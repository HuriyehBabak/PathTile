"""Training-only replay; no CUDA execution or official holdout access."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts import prototype_pathtile_shape_queries as proto
from scripts.compare_pathtile_shape_coverage import (
    coverage_queries,
    measured_policy,
    score_policy,
)

SEED = 20261023
REPEATS = 100
BUDGETS = (2, 4, 6, 8, 12, 15)

stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
out = Path("results/pathtile_selector") / f"shape_budget_replay_{stamp}"
out.mkdir(parents=True, exist_ok=False)

rows = []
splits = []
sources = {}

for gpu in ("a100", "a6000", "v100"):
    root = Path("results/pathtile_candidates") / f"{gpu}_train"

    ids, dimensions, data, _, provenance, _ = proto.load_training(root)

    if len(ids) != 20:
        raise ValueError(f"{gpu}: expected 20 training shapes")

    sources[gpu] = provenance

    # Keep validation splits independent from random-baseline sampling.
    rng_split = np.random.default_rng(SEED)
    rng_random = np.random.default_rng(SEED + 1)

    for repeat in range(REPEATS):
        validation_folds = np.array_split(
            rng_split.permutation(len(ids)),
            4,
        )

        for fold, validation in enumerate(validation_folds):
            validation = validation.tolist()

            pool = sorted(
                set(range(len(ids))) - set(validation)
            )

            # Four-fold validation leaves 15 shapes for selection.
            if len(pool) != 15:
                raise ValueError(
                    f"{gpu}, repeat={repeat}, fold={fold}: "
                    f"expected selection pool of 15 shapes, got {len(pool)}"
                )

            for budget in BUDGETS:
                if budget > len(pool):
                    raise ValueError(
                        f"Budget {budget} exceeds selection-pool size "
                        f"{len(pool)}"
                    )

                queries = {
                    "full": pool,
                    "coverage": coverage_queries(
                        pool,
                        dimensions,
                        budget,
                        proto.useful_products,
                    ),
                    "random": rng_random.choice(
                        pool,
                        budget,
                        replace=False,
                    ).tolist(),
                }

                policies = {
                    method: measured_policy(
                        data,
                        proto.PHASES,
                        {
                            phase: selected
                            for phase in proto.PHASES
                        },
                        pool,
                        len(selected),
                    )
                    for method, selected in queries.items()
                }

                for method, selected in queries.items():
                    latency, regret = score_policy(
                        data,
                        proto.PHASES,
                        policies[method],
                        validation,
                    )

                    matches_full = (
                        policies[method] == policies["full"]
                    )

                    candidate_shape_cells = (
                        len(selected)
                        * sum(
                            len(data[phase].candidates)
                            for phase in proto.PHASES
                        )
                    )

                    splits.append(
                        {
                            "gpu": gpu,
                            "repeat": repeat,
                            "fold": fold,
                            "budget": budget,
                            "method": method,
                            "selected_shapes": [
                                str(ids[i]) for i in selected
                            ],
                            "validation_shapes": [
                                str(ids[i]) for i in validation
                            ],
                            "policy": policies[method],
                            "full_policy_match": matches_full,
                            "candidate_shape_cells":
                                candidate_shape_cells,
                        }
                    )

                    for pos, latency_ms, regret_pct in zip(
                        validation,
                        latency,
                        regret,
                    ):
                        rows.append(
                            {
                                "gpu": gpu,
                                "repeat": repeat,
                                "fold": fold,
                                "budget": budget,
                                "method": method,
                                "config_id": str(ids[pos]),
                                "latency_ms": float(latency_ms),
                                "regret_pct": float(regret_pct),
                            }
                        )


per_shape = pd.DataFrame(rows)
folds = pd.DataFrame(splits)

summary = (
    per_shape
    .groupby(
        ["gpu", "budget", "method"],
        as_index=False,
    )
    .agg(
        mean_regret_pct=("regret_pct", "mean"),
        max_regret_pct=("regret_pct", "max"),
    )
)

matches = (
    folds
    .groupby(
        ["gpu", "budget", "method"],
        as_index=False,
    )
    .agg(
        full_policy_matches=("full_policy_match", "sum"),
        n_folds=("full_policy_match", "size"),
        candidate_shape_cells=("candidate_shape_cells", "mean"),
    )
)

summary = summary.merge(
    matches,
    on=["gpu", "budget", "method"],
    validate="one_to_one",
)

summary["full_policy_match_pct"] = (
    100.0
    * summary["full_policy_matches"]
    / summary["n_folds"]
)


def encode(value):
    if isinstance(value, np.generic):
        return value.item()

    raise TypeError(
        f"Unsupported JSON value: {type(value)}"
    )


code_paths = [
    Path(__file__),
    Path(proto.__file__),
    Path("scripts/compare_pathtile_shape_coverage.py"),
]

metadata = {
    "seed": SEED,
    "random_seed": SEED + 1,
    "repeats": REPEATS,
    "folds": 4,
    "shape_budgets": list(BUDGETS),
    "training_shapes_per_gpu": 20,
    "selection_shapes_per_fold": 15,
    "validation_shapes_per_fold": 5,
    "training_only": True,
    "official_holdout_read": False,
    "new_cuda_measurements": False,
    "source_provenance": sources,
    "script_sha256": {
        str(path): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in code_paths
    },
}

per_shape.to_csv(
    out / "per_shape.csv",
    index=False,
)

folds.to_csv(
    out / "fold_summary.csv",
    index=False,
)

summary.to_csv(
    out / "summary.csv",
    index=False,
)

(out / "splits_and_policies.json").write_text(
    json.dumps(
        splits,
        indent=2,
        default=encode,
    )
    + "\n"
)

(out / "metadata.json").write_text(
    json.dumps(
        metadata,
        indent=2,
        default=encode,
    )
    + "\n"
)

print(summary.to_string(index=False))

print(f"\nSaved to: {out}")

print(
    "Training-only replay complete; "
    "no measured tuning-time savings."
)