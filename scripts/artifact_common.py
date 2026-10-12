"""Shared helpers for the PathTile artifact scripts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


DEFAULT_ATOL = 1e-11
DEFAULT_RTOL = 1e-11


def resolve_project_root(value: Path | str) -> Path:
    root = Path(value).expanduser().resolve()
    if not (root / "results").is_dir():
        raise FileNotFoundError(f"Project root does not contain results/: {root}")
    return root


def read_csv(path: Path, required_columns: Iterable[str] = ()) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")
    return frame


def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array <= 0.0):
        raise ValueError(f"Invalid geometric-mean values: {array}")
    return float(np.exp(np.log(array).mean()))


def sample_std(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0:
        raise ValueError("Cannot compute a standard deviation of no values")
    return float(np.std(array, ddof=1)) if array.size > 1 else 0.0


def normalize_roundoff(value: float, tolerance: float = 1e-12) -> float:
    return 0.0 if math.isfinite(value) and abs(value) < tolerance else float(value)


def assert_frame_close(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    name: str,
    *,
    atol: float = DEFAULT_ATOL,
    rtol: float = DEFAULT_RTOL,
) -> None:
    try:
        pd.testing.assert_frame_equal(
            actual.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            atol=atol,
            rtol=rtol,
        )
    except AssertionError as exc:
        raise AssertionError(f"Artifact mismatch for {name}:\n{exc}") from exc


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
