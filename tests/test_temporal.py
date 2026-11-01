import ast
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from dataset_temporal import TemporalDataset, build_datasets, forecast_metrics


def series(n=60, building=0, meter=0, start="2016-01-01", offset=0):
    return pd.DataFrame({"building_id": building, "meter": meter,
        "timestamp": pd.date_range(start, periods=n, freq="h"),
        "meter_reading": np.arange(n, dtype=float) + offset,
        "air_temperature": np.arange(n, dtype=float),
        "cloud_coverage": 2., "dew_temperature": 3.})


class DatasetTests(unittest.TestCase):
    def test_exact_window_and_target(self):
        data = TemporalDataset(series(), fraction=1)
        self.assertEqual(len(data), 12)
        first = data[0]
        self.assertEqual(first["input_sequence"].shape, (48, 4))
        self.assertAlmostEqual(float(first["label_sequence"]), np.log1p(48), places=6)
        restored = first["input_sequence"] * data.scaler["std"] + data.scaler["mean"]
        np.testing.assert_allclose(restored[:, 0], np.log1p(np.arange(48)), atol=1e-6)

    def test_shuffled_multi_series_never_crosses(self):
        frame = pd.concat([series(), series(building=1, offset=100),
                           series(meter=1, offset=200)], ignore_index=True).sample(frac=1, random_state=7)
        data = TemporalDataset(frame, fraction=1)
        self.assertEqual(len(data), 36)
        expected = np.r_[np.arange(48, 60), np.arange(248, 260), np.arange(148, 160)]
        np.testing.assert_allclose(np.expm1(data.evaluation_targets()), expected)

    def test_gap_and_invalid_row_break_windows(self):
        frame = series(110)
        frame.loc[50, "air_temperature"] = np.nan
        data = TemporalDataset(frame, fraction=1)
        # Targets 48,49 and 99..109 only; all edges across missing hour rejected.
        np.testing.assert_array_equal(data.target_source_rows, np.r_[48, 49, np.arange(99, 110)])
        self.assertEqual(data.audit["dropped_invalid_rows"], 1)

    def test_duplicate_is_error(self):
        frame = pd.concat([series(), series().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "duplicate"):
            TemporalDataset(frame)

    def test_train_scaler_reused_without_refit(self):
        train = TemporalDataset(series(), fraction=1)
        val = TemporalDataset(series(offset=1000), fraction=1, split="val", scaler=train.scaler)
        self.assertIs(val.scaler, train.scaler)
        self.assertTrue(np.isfinite(val.inputs).all())
        self.assertNotAlmostEqual(float(val.inputs[:, 0].mean()), 0)
        with self.assertRaisesRegex(ValueError, "training scaler"):
            TemporalDataset(series(), split="test")

    def test_sampling_complete_windows_is_reproducible(self):
        first = TemporalDataset(series(1000), fraction=.1)
        second = TemporalDataset(series(1000), fraction=.1)
        self.assertEqual(len(first), 95)
        np.testing.assert_array_equal(first.target_source_rows, second.target_source_rows)
        self.assertEqual(len(np.unique(first.target_indices)), len(first))

    def test_exact_minimum_and_empty(self):
        self.assertEqual(len(TemporalDataset(series(49), fraction=1)), 1)
        with self.assertRaisesRegex(ValueError, "no complete"):
            TemporalDataset(series(48))

    def test_negative_energy_breaks_window(self):
        frame = series(110)
        frame.loc[50, "meter_reading"] = -1
        data = TemporalDataset(frame, fraction=1)
        np.testing.assert_array_equal(data.target_source_rows, np.r_[48, 49, np.arange(99, 110)])

    def test_split_chronology_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            for split, start in zip(("train", "val", "test"),
                                    ("2016-01-01", "2016-02-01", "2016-03-01")):
                series(start=start).to_csv(Path(directory) / f"{split}.csv", index=False)
            datasets, manifest = build_datasets(directory, fraction=1)
            self.assertEqual(manifest["splits"]["test"]["selected_windows"], 12)
            self.assertEqual(len(manifest["splits"]["train"]["file_sha256"]), 64)
            self.assertEqual(datasets["val"].scaler, datasets["train"].scaler)
            series().to_csv(Path(directory) / "val.csv", index=False)
            with self.assertRaisesRegex(ValueError, "overlaps"):
                build_datasets(directory)


class MetricTests(unittest.TestCase):
    def test_log_computation_matches_raw_energy_rmsle(self):
        target = np.array([0., 5., 17., 100.])
        predicted_log = np.array([-2., .3, 3., 5.])
        raw_prediction = np.expm1(np.maximum(predicted_log, 0))
        expected = np.sqrt(np.mean((np.log1p(raw_prediction)-np.log1p(target))**2))
        result = forecast_metrics(predicted_log, np.log1p(target), np.log1p(target))
        self.assertAlmostEqual(result["test_rmsle"], expected)
        self.assertEqual(result["persistence_baseline_rmsle"], 0)
        self.assertEqual(result["prediction_nonpositive_fraction"], .25)

    def test_zero_baseline_and_nonfinite(self):
        target = np.log1p([1., 2., 3.])
        result = forecast_metrics(np.zeros(3), target, target)
        self.assertEqual(result["test_rmsle"], result["zero_baseline_rmsle"])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            forecast_metrics([np.nan]*3, target, target)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch unavailable in this environment")
    def test_actual_loss_has_gradient_for_negative_prediction(self):
        # Extract the exact production loss without importing the CUDA runtime.
        import torch
        path = Path(__file__).resolve().parents[1] / "kernels/warp_tiled_ampere_temporal.py"
        tree = ast.parse(path.read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "log_mse")
        scope = {"torch": torch, "F": torch.nn.functional}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), scope)
        prediction = torch.tensor([-2.], requires_grad=True)
        loss = scope["log_mse"](prediction, torch.tensor([1.]))
        loss.backward()
        self.assertAlmostEqual(prediction.grad.item(), -6.)
        data = TemporalDataset(series(), fraction=1)
        batch = next(iter(torch.utils.data.DataLoader(data, batch_size=4)))
        self.assertEqual(tuple(batch["input_sequence"].shape), (4, 48, 4))
        self.assertEqual(batch["label_sequence"].dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
