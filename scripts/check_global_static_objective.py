#!/usr/bin/env python3

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.aggregate_pathtile_results import (
    GPU_CONFIGS,
    aggregate_candidate_root,
    _selected_phase_time,
)


training = {}

for gpu_label, gpu_key, _ in GPU_CONFIGS:
    _, candidates, _, combined = aggregate_candidate_root(
        ROOT,
        Path(f"results/pathtile_candidates/{gpu_key}_train"),
    )
    training[gpu_label] = (candidates, combined)


def common_values(phase: str) -> list[int]:
    sets = []

    for candidates, _ in training.values():
        values = set(
            candidates.loc[
                candidates["phase"] == phase,
                "candidate_value",
            ].astype(int)
        )
        sets.append(values)

    return sorted(set.intersection(*sets))


fwd_values = common_values("fwd")
dx_values = common_values("bwd_input")
dk_values = common_values("bwd_weight")

print("Common candidate values")
print("FWD:", fwd_values)
print("dX :", dx_values)
print("dK :", dk_values)

rows = []

for fwd, dx, dk in itertools.product(
    fwd_values,
    dx_values,
    dk_values,
):
    ratios = []
    raw_total_ms = 0.0

    for gpu_label, (candidates, combined) in training.items():
        for oracle_row in combined.itertuples(index=False):
            config_id = oracle_row.config_id

            selected_total = (
                _selected_phase_time(
                    candidates,
                    config_id,
                    "fwd",
                    fwd,
                )
                + _selected_phase_time(
                    candidates,
                    config_id,
                    "bwd_input",
                    dx,
                )
                + _selected_phase_time(
                    candidates,
                    config_id,
                    "bwd_weight",
                    dk,
                )
            )

            oracle_total = float(
                oracle_row.oracle_total_ms
            )

            raw_total_ms += selected_total
            ratios.append(
                selected_total / oracle_total
            )

    ratios = np.asarray(ratios, dtype=float)

    geomean_ratio = float(
        np.exp(np.mean(np.log(ratios)))
    )

    rows.append(
        {
            "fwd_warps": fwd,
            "dx_warps": dx,
            "dk_block": dk,
            "raw_total_ms": raw_total_ms,
            "mean_regret_pct": float(
                np.mean((ratios - 1.0) * 100.0)
            ),
            "geomean_regret_pct": (
                geomean_ratio - 1.0
            ) * 100.0,
            "max_regret_pct": float(
                np.max((ratios - 1.0) * 100.0)
            ),
        }
    )


result = pd.DataFrame(rows)

print("\nBest by raw summed latency:")
print(
    result.sort_values(
        ["raw_total_ms", "fwd_warps", "dx_warps", "dk_block"]
    )
    .head(10)
    .to_string(index=False)
)

print("\nBest by geometric-mean normalized regret:")
print(
    result.sort_values(
        [
            "geomean_regret_pct",
            "fwd_warps",
            "dx_warps",
            "dk_block",
        ]
    )
    .head(10)
    .to_string(index=False)
)

out = (
    ROOT
    / "results/pathtile_selector/selector_baselines_v1/"
      "global_static_objective_sensitivity.csv"
)

result.to_csv(out, index=False)

print("\nSaved:")
print(out)
