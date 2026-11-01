"""Inspect one corrected Warp/PyTorch pair; no exact-equality training assertion."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def review(root):
    manifests, metrics, predictions, times = {}, {}, {}, {}
    for backend in ("warp", "pytorch"):
        folder = Path(root) / backend
        manifests[backend] = json.loads((folder / "training_manifest.json").read_text())
        if manifests[backend]["backend"] != backend:
            raise ValueError("Backend metadata mismatch")
        if manifests[backend]["protocol"] != "temporal_logmse_trainmedian_v2":
            raise ValueError("Unexpected protocol")
        metrics[backend] = pd.read_csv(folder / "training_final_test_metrics.csv").iloc[0]
        with np.load(folder / "test_predictions.npz") as data:
            predictions[backend] = {key: data[key].copy() for key in data.files}
        times[backend] = pd.read_csv(folder / "training_epoch_system_summary.csv")
    for key in ("config", "scaler", "splits", "initial_model_sha256", "source_sha256"):
        if manifests["warp"][key] != manifests["pytorch"][key]:
            raise ValueError(f"Pair mismatch in {key}")
    for key in ("target_source_row", "target_log", "persistence_log"):
        np.testing.assert_array_equal(predictions["warp"][key], predictions["pytorch"][key])
    np.testing.assert_array_equal(times["warp"].epoch, times["pytorch"].epoch)
    for backend in ("warp", "pytorch"):
        metric = metrics[backend]
        prediction = predictions[backend]["predicted_log"]
        if not np.isfinite(prediction).all():
            raise ValueError("Nonfinite prediction")
        timing = times[backend].train_wall_time_s.to_numpy()
        if not np.isfinite(timing).all() or (timing <= 0).any():
            raise ValueError("Invalid epoch timing")
        print(backend, json.dumps({
            "test_rmsle": float(metric.test_rmsle),
            "zero_baseline_rmsle": float(metric.zero_baseline_rmsle),
            "persistence_baseline_rmsle": float(metric.persistence_baseline_rmsle),
            "beats_zero": bool(metric.test_rmsle < metric.zero_baseline_rmsle),
            "beats_persistence": bool(metric.test_rmsle < metric.persistence_baseline_rmsle),
            "prediction_nonpositive_fraction": float(metric.prediction_nonpositive_fraction),
            "parameter_change_l2": float(metric.parameter_change_l2),
        }, indent=2))
    difference = predictions["warp"]["predicted_log"].astype(float) - predictions["pytorch"]["predicted_log"].astype(float)
    print(json.dumps({
        "paired_rmsle_absolute_difference": abs(float(metrics["warp"].test_rmsle) - float(metrics["pytorch"].test_rmsle)),
        "paired_log_prediction_max_abs_difference": float(np.max(np.abs(difference))),
        "paired_log_prediction_rmse_difference": float(np.sqrt(np.mean(difference**2))),
        "epoch_speedup_pytorch_over_warp": float(times["pytorch"].train_wall_time_s.mean() / times["warp"].train_wall_time_s.mean()),
    }, indent=2))
    print("Inspect training_history.csv and correctness.csv too. This report does not establish numerical equivalence or convergence.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pair_directory", type=Path)
    review(parser.parse_args().pair_directory)
