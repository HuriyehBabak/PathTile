#!/usr/bin/env python3
"""Generate ACM TACO-ready figures and tables for the PathTile study.

This script performs no benchmarking. It reads only explicitly selected final
result files, validates their schemas, and writes vector PDF figures, 600-dpi
PNG previews, CSV tables, and booktabs-compatible LaTeX tables.

Typical use from the project root:

    python paper/generate_final_figures.py --project-root .
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import tempfile
from typing import Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "pathtile-matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

# ACM TACO uses the ACM two-column article layout.  These dimensions target
# the full text width; all labels remain at or above 7 pt after inclusion at
# 100% scale.  Keep captions in LaTeX, not inside the figure files.
ACM_TEXT_WIDTH_IN = 7.0
ACM_COLUMN_WIDTH_IN = 3.35
RASTER_DPI = 600

GPU_ORDER = [
    "NVIDIA A100-PCIE-40GB",
    "NVIDIA RTX A6000",
    "Tesla V100-SXM3-32GB",
]
GPU_SHORT = {
    "NVIDIA A100-PCIE-40GB": "A100",
    "NVIDIA RTX A6000": "RTX A6000",
    "Tesla V100-SXM3-32GB": "V100",
}
# Internal method identifiers used by the kernel benchmark CSV files.
# Keep these values unchanged because they are part of the result schema.
METHOD_ORDER = ["PyTorch", "Naive", "GMC", "Shared", "Warp"]

# Publication labels for kernel-design results.  PathTile is not a fifth
# kernel design: it is the offline policy that selects launch configurations
# for the warp-tiled kernels.
KERNEL_METHOD_LABELS = {
    "PyTorch": "PyTorch",
    "Naive": "Naive",
    "GMC": "GMC",
    "Shared": "Shared",
    "Warp": "Warp-tiled",
}

# These external-baseline rows are explicitly produced for the final PathTile
# system, so PathTile is the correct publication label here.
EXTERNAL_METHOD_ORDER = ["PyTorch", "Triton", "TVM-TIRx", "PathTile"]

# Internal end-to-end backend identifiers.  The ``warp`` backend executes the
# final PathTile-selected warp-tiled configuration in the system experiment.
BACKEND_ORDER = ["pytorch", "warp"]
SYSTEM_BACKEND_LABELS = {
    "pytorch": "PyTorch",
    "warp": "PathTile",
}
PHASE_COLUMNS = [
    ("FWD", "fwd_mean_ms"),
    ("dX", "bwd_input_mean_ms"),
    ("dK", "bwd_weight_mean_ms"),
]
# ---------------------------------------------------------------------------
# Publication palette
# ---------------------------------------------------------------------------
# A muted, journal-oriented palette is used throughout the paper. Color has
# one semantic role at a time: method identity, execution phase, GPU identity,
# or end-to-end measurement scope. Strong saturation is intentionally avoided;
# the brick-red accent is reserved for PathTile / the final warp configuration
# when method identity is the quantity being encoded.
EDGE_COLOR = "#4A4A4A"
TEXT_COLOR = "#2F2F2F"
GRID_COLOR = "#E6E6E6"

# ------------------------------------------------------------------
# Publication palette
# ------------------------------------------------------------------
# Clean, conventional Tableau-style colors: clearly distinguishable,
# visually balanced, and appropriate for journal figures.
COLORS = {
    "PyTorch": "#6B6B6B",      # dark gray
    "Naive": "#BAB0AC",        # light gray
    "GMC": "#4E79A7",          # blue
    "Shared": "#59A14F",       # green
    "Warp": "#E15759",         # red
    "Triton": "#B07AA1",       # purple
    "TVM-TIRx": "#F28E2B",     # orange
    "PathTile": "#E15759",     # primary accent
    "pytorch": "#6B6B6B",
    "warp": "#E15759",
}

# Computational paths: strong, familiar distinctions.
PHASE_COLORS = {
    "FWD": "#4E79A7",          # blue
    "dX": "#59A14F",           # green
    "dK": "#F28E2B",           # orange
    "Other GPU": "#BAB0AC",    # gray
}

# Kept only for compatibility; hatches are intentionally not used in the
# final color figures because they make the plots visually noisy.
PHASE_HATCHES = {
    "FWD": "",
    "dX": "",
    "dK": "",
    "Other GPU": "",
}

# GPU identity.
GPU_COLORS = {
    "A100": "#4E79A7",         # blue
    "A6000": "#E15759",        # red
    "RTX A6000": "#E15759",
    "V100": "#59A14F",         # green
    "All": "#222222",
}

# End-to-end measurement scopes.
E2E_COLORS = {
    "Model FWD": "#4E79A7",
    "Backward": "#B07AA1",
    "GPU stages": "#4A4A4A",
    "Epoch wall": "#8C939A",
    "Other GPU": "#BAB0AC",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path.cwd(),
        help="Project root containing results/ (default: current directory).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/final_artifacts_TACO"),
        help="Output directory, relative to project root unless absolute.",
    )
    return parser.parse_args()



def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            # DejaVu Sans ships with Matplotlib, making the output reproducible
            # on the cluster and on artifact-evaluation machines.
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 7.0,
            "ytick.labelsize": 7.0,
            "legend.fontsize": 7.0,
            "axes.linewidth": 0.60,
            "axes.edgecolor": EDGE_COLOR,
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "text.color": TEXT_COLOR,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.38,
            "grid.alpha": 0.70,
            "grid.linestyle": "-",
            "lines.linewidth": 1.2,
            "hatch.linewidth": 0.65,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.major.width": 0.5,
            "ytick.major.width": 0.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": True,
            "savefig.dpi": RASTER_DPI,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
            "savefig.transparent": False,
        }
    )



def style_axis(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    """Apply the shared publication style to a conventional Cartesian axis."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)



def read_csv(root: Path, relative_path: str) -> pd.DataFrame:
    path = root / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Required canonical result is missing: {path}")
    return pd.read_csv(path)



def require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")



def geometric_mean(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    if array.size == 0 or not np.isfinite(array).all() or np.any(array <= 0):
        raise ValueError(f"Invalid geometric-mean values: {array}")
    return float(np.exp(np.log(array).mean()))



def ordered_groups(values: Iterable[str]) -> list[str]:
    preferred = [
        "batch",
        "channel",
        "sequence",
        "length",
        "kernel",
        "main",
        "holdout",
    ]
    unique = list(dict.fromkeys(str(value) for value in values))
    rank = {name: index for index, name in enumerate(preferred)}
    return sorted(unique, key=lambda item: (rank.get(item, len(rank)), item))



def save_figure(fig: plt.Figure, stem: Path) -> list[Path]:
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    pdf_metadata = {
        "Creator": "PathTile paper artifact generator",
        "Title": stem.name,
    }
    fig.savefig(paths[0], metadata=pdf_metadata)
    fig.savefig(paths[1], dpi=RASTER_DPI)
    plt.close(fig)
    return paths



def write_table(frame: pd.DataFrame, stem: Path) -> list[Path]:
    csv_path = stem.with_suffix(".csv")
    tex_path = stem.with_suffix(".tex")
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(dataframe_to_latex(frame), encoding="utf-8")
    return [csv_path, tex_path]



def latex_escape(value: object) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in str(value))



def latex_cell(value: object) -> str:
    if pd.isna(value):
        return "--"
    if isinstance(value, (bool, np.bool_)):
        return "True" if bool(value) else "False"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.3f}"
    return latex_escape(value)



def dataframe_to_latex(frame: pd.DataFrame) -> str:
    alignment = "".join(
        "r" if pd.api.types.is_numeric_dtype(frame[column]) else "l"
        for column in frame.columns
    )
    header = " & ".join(latex_escape(column) for column in frame.columns)
    rows = [
        " & ".join(latex_cell(value) for value in row)
        for row in frame.itertuples(index=False, name=None)
    ]
    lines = [
        rf"\begin{{tabular}}{{{alignment}}}",
        r"\toprule",
        header + r" \\",
        r"\midrule",
        *(row + r" \\" for row in rows),
        r"\bottomrule",
        r"\end{tabular}",
        "",
    ]
    return "\n".join(lines)


def cross_gpu_figure_detailed(
    cross: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Generate the publication-ready cross-GPU performance figure."""

    # ------------------------------------------------------------------
    # Select PyTorch and the final warp-tiled kernel.
    # ------------------------------------------------------------------
    selected = cross[cross["method"].isin(["PyTorch", "Warp"])].copy()

    selected["gpu"] = pd.Categorical(
        selected["gpu"],
        categories=GPU_ORDER,
        ordered=True,
    )
    selected = selected.sort_values(["gpu", "method"])

    expected_rows = len(GPU_ORDER) * 2
    if len(selected) != expected_rows:
        raise ValueError(
            "Expected one PyTorch and one Warp result for each GPU."
        )

    # ------------------------------------------------------------------
    # Full ACM two-column figure.
    # ------------------------------------------------------------------
    fig, (ax_latency, ax_speedup) = plt.subplots(
        1,
        2,
        figsize=(ACM_TEXT_WIDTH_IN, 2.70),
        gridspec_kw={
            "width_ratios": [1.35, 0.90],
            "wspace": 0.34,
        },
    )

    # ================================================================
    # (a) Absolute latency and path breakdown
    # ================================================================

    group_centers = np.arange(len(GPU_ORDER), dtype=float) * 2.15

    bar_offset = 0.34
    bar_width = 0.56

    method_specs = [
        ("PyTorch", -bar_offset),
        ("Warp", +bar_offset),
    ]

    for method, offset in method_specs:
        rows = (
            selected[selected["method"] == method]
            .set_index("gpu")
            .loc[GPU_ORDER]
        )

        positions = group_centers + offset
        bottom = np.zeros(len(GPU_ORDER), dtype=float)

        for phase, column in PHASE_COLUMNS:
            phase_values = rows[column].to_numpy(dtype=float)

            ax_latency.bar(
                positions,
                phase_values,
                width=bar_width,
                bottom=bottom,
                color=PHASE_COLORS[phase],
                edgecolor=EDGE_COLOR,
                linewidth=0.4,
                zorder=3,
            )

            bottom += phase_values

        errors = (
            rows["total_between_trial_std_ms"]
            .fillna(0.0)
            .to_numpy(dtype=float)
        )

        ax_latency.errorbar(
            positions,
            bottom,
            yerr=errors,
            fmt="none",
            ecolor=EDGE_COLOR,
            elinewidth=0.65,
            capsize=1.8,
            capthick=0.65,
            zorder=5,
        )

    # ------------------------------------------------------------------
    # Direct implementation labels.
    # ------------------------------------------------------------------
    bar_positions = []
    bar_labels = []

    for center in group_centers:
        bar_positions.extend(
            [
                center - bar_offset,
                center + bar_offset,
            ]
        )

        bar_labels.extend(
            [
                "PyTorch",
                KERNEL_METHOD_LABELS["Warp"],
            ]
        )

    ax_latency.set_xticks(bar_positions)

    ax_latency.set_xticklabels(
        bar_labels,
        fontsize=6.3,
        rotation=22,
        ha="right",
    )

    ax_latency.tick_params(
        axis="x",
        length=0,
        pad=2,
    )

    # GPU group labels.
    for center, gpu in zip(
        group_centers,
        GPU_ORDER,
        strict=True,
    ):
        ax_latency.text(
            center,
            -0.20,
            GPU_SHORT[gpu],
            transform=ax_latency.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=7.0,
            fontweight="bold",
            clip_on=False,
        )

    # Very light separators between GPU groups.
    separators = (
        group_centers[:-1] + group_centers[1:]
    ) / 2.0

    for separator in separators:
        ax_latency.axvline(
            separator,
            color=GRID_COLOR,
            linewidth=0.45,
            zorder=0,
        )

    ymax = float(selected["total_mean_ms"].max())

    ax_latency.set_ylabel("Latency (ms)")
    ax_latency.set_ylim(0.0, ymax * 1.07)

    ax_latency.set_title(
        "(a)",
        loc="left",
        pad=2,
        fontsize=8.0,
        fontweight="bold",
    )

    style_axis(ax_latency)

    # ================================================================
    # Shared phase legend
    # ================================================================

    phase_handles = [
        Patch(
            facecolor=PHASE_COLORS[phase],
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
            label=phase,
        )
        for phase, _ in PHASE_COLUMNS
    ]

    fig.legend(
        handles=phase_handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.985),
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.4,
        borderaxespad=0.0,
    )

    # ================================================================
    # (b) Path-specific and aggregate speedup
    # ================================================================

    warp_tiled_rows = (
        selected[selected["method"] == "Warp"]
        .set_index("gpu")
        .loc[GPU_ORDER]
    )

    speedup_rows = [
        ("FWD", "fwd_speedup_vs_pytorch"),
        ("dX", "bwd_input_speedup_vs_pytorch"),
        ("dK", "bwd_weight_speedup_vs_pytorch"),
        ("Total", "total_speedup_vs_pytorch"),
    ]

    matrix = np.vstack(
        [
            warp_tiled_rows[column].to_numpy(dtype=float)
            for _, column in speedup_rows
        ]
    )

    # ------------------------------------------------------------------
    # One sequential heatmap only.
    # ------------------------------------------------------------------
    image = ax_speedup.imshow(
        matrix,
        cmap="viridis",
        vmin=1.0,
        vmax=7.5,
        aspect="auto",
        interpolation="nearest",
    )

    # Exact values inside cells.
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]

            rgba = image.cmap(image.norm(value))

            luminance = (
                0.2126 * rgba[0]
                + 0.7152 * rgba[1]
                + 0.0722 * rgba[2]
            )

            text_color = "black" if luminance > 0.55 else "white"

            ax_speedup.text(
                column_index,
                row_index,
                f"{value:.2f}×",
                ha="center",
                va="center",
                color=text_color,
                fontsize=7.0,
                fontweight=(
                    "bold"
                    if row_index == matrix.shape[0] - 1
                    else "normal"
                ),
            )

    # Separate aggregate result from individual paths.
    ax_speedup.axhline(
        matrix.shape[0] - 1.5,
        color="white",
        linewidth=1.5,
    )

    ax_speedup.set_xticks(
        range(len(GPU_ORDER)),
        [GPU_SHORT[gpu] for gpu in GPU_ORDER],
    )

    ax_speedup.set_yticks(
        range(len(speedup_rows)),
        [label for label, _ in speedup_rows],
    )

    # Emphasize the aggregate row structurally.
    ax_speedup.get_yticklabels()[-1].set_fontweight("bold")

    ax_speedup.tick_params(
        axis="both",
        length=0,
    )

    ax_speedup.set_title(
        "(b)",
        loc="left",
        pad=2,
        fontsize=8.0,
        fontweight="bold",
    )

    for spine in ax_speedup.spines.values():
        spine.set_visible(False)

    # ================================================================
    # Single colorbar
    # ================================================================

    colorbar = fig.colorbar(
        image,
        ax=ax_speedup,
        fraction=0.046,
        pad=0.030,
    )

    colorbar.set_label(
        "Speedup (×)",
        labelpad=2,
    )

    colorbar.set_ticks(
        [1, 2, 3, 4, 5, 6, 7]
    )

    colorbar.outline.set_linewidth(0.5)

    colorbar.ax.tick_params(
        labelsize=7.0,
        length=2.5,
        width=0.5,
    )

    # ================================================================
    # Final layout
    # ================================================================

    fig.subplots_adjust(
        left=0.075,
        right=0.965,
        top=0.86,
        bottom=0.25,
        wspace=0.34,
    )

    return save_figure(
        fig,
        figures_dir / "fig_cross_gpu_latency",
    )

def v100_ablation_figure(
    cross: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Generate the V100 cumulative optimization ablation figure."""

    v100 = (
        cross[cross["gpu"] == GPU_ORDER[2]]
        .copy()
        .set_index("method")
        .loc[METHOD_ORDER]
    )

    values = v100["total_mean_ms"].to_numpy(dtype=float)

    errors = (
        v100["total_between_trial_std_ms"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    fig, (ax, ax_gain) = plt.subplots(
        1,
        2,
        figsize=(ACM_TEXT_WIDTH_IN, 2.45),
        gridspec_kw={
            "width_ratios": [1.22, 1.0],
            "wspace": 0.40,
        },
    )

    # ================================================================
    # (a) Optimization progression
    # ================================================================

    x = np.arange(len(METHOD_ORDER), dtype=float)
    bottom = np.zeros(len(METHOD_ORDER), dtype=float)

    # Keep the same semantic phase colors used throughout the paper:
    # FWD = blue, dX = green, dK = orange.
    for phase, column in PHASE_COLUMNS:
        phase_values = v100[column].to_numpy(dtype=float)

        ax.bar(
            x,
            phase_values,
            bottom=bottom,
            width=0.68,
            color=PHASE_COLORS[phase],
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
            label=phase,
            zorder=3,
        )

        bottom += phase_values

    ax.errorbar(
        x,
        bottom,
        yerr=errors,
        fmt="none",
        ecolor=EDGE_COLOR,
        elinewidth=0.65,
        capsize=1.8,
        capthick=0.65,
        zorder=5,
    )

    display_methods = [KERNEL_METHOD_LABELS[method] for method in METHOD_ORDER]

    ax.set_xticks(
        x,
        display_methods,
        rotation=18,
        ha="right",
    )

    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0.0, values.max() * 1.08)

    # Same style as cross-GPU figure: panel label only.
    ax.set_title(
        "(a)",
        loc="left",
        pad=2,
        fontsize=8.0,
        fontweight="bold",
    )

    style_axis(ax)

    # ================================================================
    # Shared phase legend
    # ================================================================

    phase_handles = [
        Patch(
            facecolor=PHASE_COLORS[phase],
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
            label=phase,
        )
        for phase, _ in PHASE_COLUMNS
    ]

    fig.legend(
        handles=phase_handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.4,
        borderaxespad=0.0,
    )

    # ================================================================
    # (b) Incremental latency reduction
    # ================================================================

    savings = values[:-1] - values[1:]

    if np.any(savings <= 0):
        raise ValueError(
            "V100 optimization progression is expected to improve monotonically"
        )

    transitions = [
        "PyTorch → Naive",
        "Naive → GMC",
        "GMC → Shared",
        "Shared → Warp-tiled",
    ]

    y = np.arange(len(transitions), dtype=float)

    # Neutral color because these bars represent optimization steps,
    # not FWD/dX/dK.
    ax_gain.barh(
        y,
        savings,
        color="#8F969B",
        edgecolor=EDGE_COLOR,
        linewidth=0.4,
        height=0.62,
        zorder=3,
    )

    # Exact contribution of each optimization step.
    for ypos, saving in zip(y, savings, strict=True):
        ax_gain.text(
            saving + savings.max() * 0.025,
            ypos,
            f"{saving:.1f} ms",
            va="center",
            ha="left",
            fontsize=6.8,
        )

    ax_gain.set_yticks(
        y,
        transitions,
    )

    ax_gain.invert_yaxis()

    ax_gain.set_xlabel("Latency removed (ms)")
    ax_gain.set_xlim(
        0.0,
        savings.max() * 1.25,
    )

    # Same style as cross-GPU figure: panel label only.
    ax_gain.set_title(
        "(b)",
        loc="left",
        pad=2,
        fontsize=8.0,
        fontweight="bold",
    )

    style_axis(
        ax_gain,
        grid_axis="x",
    )

    # ================================================================
    # Final layout
    # ================================================================

    fig.tight_layout(
        rect=(0.01, 0.01, 0.99, 0.90),
    )

    return save_figure(
        fig,
        figures_dir / "fig_v100_ablation",
    )

def v100_ablation_figure_detailed(
    cross: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Generate the publication-ready V100 optimization ablation figure."""

    v100 = (
        cross[cross["gpu"] == GPU_ORDER[2]]
        .copy()
        .set_index("method")
        .loc[METHOD_ORDER]
    )

    values = v100["total_mean_ms"].to_numpy(dtype=float)

    errors = (
        v100["total_between_trial_std_ms"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    fig, (ax, ax_gain) = plt.subplots(
        1,
        2,
        figsize=(ACM_TEXT_WIDTH_IN, 2.45),
        gridspec_kw={
            "width_ratios": [1.08, 1.0],   # left panel a bit narrower
            "wspace": 0.58,                # more gap between panels
        },
    )

    # ================================================================
    # (a) Cumulative optimization progression
    # ================================================================
    x = np.arange(len(METHOD_ORDER), dtype=float)
    bottom = np.zeros(len(METHOD_ORDER), dtype=float)

    for phase, column in PHASE_COLUMNS:
        phase_values = v100[column].to_numpy(dtype=float)

        ax.bar(
            x,
            phase_values,
            bottom=bottom,
            width=0.68,
            color=PHASE_COLORS[phase],
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
            label=phase,
            zorder=3,
        )

        bottom += phase_values

    ax.errorbar(
        x,
        bottom,
        yerr=errors,
        fmt="none",
        ecolor=EDGE_COLOR,
        elinewidth=0.65,
        capsize=1.8,
        capthick=0.65,
        zorder=5,
    )

    display_methods = [
        "PyTorch",
        "Naive",
        "GMC",
        "Shared",
        "Warp-tiled",
    ]

    ax.set_xticks(
        x,
        display_methods,
        rotation=18,
        ha="right",
    )

    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0.0, values.max() * 1.08)

    ax.set_title(
        "(a)",
        loc="left",
        pad=2,
        fontsize=8.0,
        fontweight="bold",
    )

    style_axis(ax)

    # ================================================================
    # Shared phase legend
    # ================================================================
    phase_handles = [
        Patch(
            facecolor=PHASE_COLORS[phase],
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
            label=phase,
        )
        for phase, _ in PHASE_COLUMNS
    ]

    fig.legend(
        handles=phase_handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.4,
        borderaxespad=0.0,
    )

    # ================================================================
    # (b) Incremental contribution of each optimization stage
    # ================================================================
    savings = values[:-1] - values[1:]

    if np.any(savings <= 0):
        raise ValueError(
            "V100 optimization progression is expected to improve monotonically"
        )

    transitions = [
        "PyTorch → Naive",
        "Naive → GMC",
        "GMC → Shared",
        "Shared → Warp-tiled",
    ]

    y = np.arange(len(transitions), dtype=float)

    ax_gain.barh(
        y,
        savings,
        color="#8F969B",
        edgecolor=EDGE_COLOR,
        linewidth=0.4,
        height=0.62,
        zorder=3,
    )

    for ypos, saving in zip(y, savings, strict=True):
        ax_gain.text(
            saving + savings.max() * 0.025,
            ypos,
            f"{saving:.1f} ms",
            ha="left",
            va="center",
            fontsize=6.8,
        )

    ax_gain.set_yticks(y, transitions)
    ax_gain.invert_yaxis()

    # important: smaller y-labels + a bit more padding
    ax_gain.tick_params(
        axis="y",
        labelsize=6.6,
        pad=4,
    )

    ax_gain.set_xlabel("Latency removed (ms)")
    ax_gain.set_xlim(0.0, savings.max() * 1.30)

    ax_gain.set_title(
        "(b)",
        loc="left",
        pad=2,
        fontsize=8.0,
        fontweight="bold",
    )

    style_axis(
        ax_gain,
        grid_axis="x",
    )

    # reserve space for legend
    fig.tight_layout(
        rect=(0.01, 0.01, 0.99, 0.90),
        w_pad=1.2,
    )

    return save_figure(
        fig,
        figures_dir / "fig_v100_ablation",
    )

def external_baseline_figure_detailed(
    external: pd.DataFrame,
    external_paper: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Compare PathTile with PyTorch and external compiler baselines."""

    # ------------------------------------------------------------------
    # Validate input data.
    # ------------------------------------------------------------------
    require_columns(
        external,
        [
            "gpu",
            "method",
            "fwd_mean_ms",
            "bwd_input_mean_ms",
            "bwd_weight_mean_ms",
            "total_mean_ms",
            "total_between_trial_std_ms",
            "evidence_status",
        ],
        "External compiler baselines",
    )

    require_columns(
        external_paper,
        [
            "gpu",
            "tvm_evidence_status",
        ],
        "External compiler paper table",
    )

    gpu_order = [
        "A100",
        "RTX A6000",
        "V100",
    ]

    expected_methods = {
        "A100": set(EXTERNAL_METHOD_ORDER),
        "RTX A6000": set(EXTERNAL_METHOD_ORDER),
        "V100": {"PyTorch", "TVM-TIRx", "PathTile"},
    }

    if external.duplicated(["gpu", "method"]).any():
        raise ValueError(
            "External baseline table contains duplicate GPU/method rows"
        )

    for gpu in gpu_order:
        found = set(
            external.loc[
                external["gpu"] == gpu,
                "method",
            ]
        )

        if found != expected_methods[gpu]:
            raise ValueError(
                f"Unexpected external-baseline methods for "
                f"{gpu}: {sorted(found)}"
            )

    paper_index = (
        external_paper
        .set_index("gpu")
        .loc[gpu_order]
    )

    expected_tvm_status = [
        "PASS",
        "PASS",
        "QUALIFIED_WITH_CAVEAT",
    ]

    if (
        paper_index["tvm_evidence_status"].tolist()
        != expected_tvm_status
    ):
        raise ValueError(
            "Unexpected TVM evidence status in paper table"
        )

    # ------------------------------------------------------------------
    # Figure layout.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(ACM_TEXT_WIDTH_IN, 2.55),
        sharey=True,
        gridspec_kw={
            "wspace": 0.22,
        },
    )

    x = np.arange(
        len(EXTERNAL_METHOD_ORDER),
        dtype=float,
    )

    ymax = float(
        external["total_mean_ms"].max()
    )

    base_tick_labels = [
        "PyTorch",
        "Triton",
        "TVM",
        "PathTile",
    ]

    # ================================================================
    # Three GPU panels.
    # ================================================================
    for panel_index, (ax, gpu) in enumerate(
        zip(axes, gpu_order, strict=True)
    ):
        rows = (
            external[external["gpu"] == gpu]
            .set_index("method")
        )

        valid = np.asarray(
            [
                method in rows.index
                for method in EXTERNAL_METHOD_ORDER
            ],
            dtype=bool,
        )

        totals = np.full(
            len(EXTERNAL_METHOD_ORDER),
            np.nan,
            dtype=float,
        )

        errors = np.zeros(
            len(EXTERNAL_METHOD_ORDER),
            dtype=float,
        )

        bottom = np.zeros(
            len(EXTERNAL_METHOD_ORDER),
            dtype=float,
        )

        # --------------------------------------------------------------
        # Stacked FWD / dX / dK latency.
        #
        # Publication-wide semantic encoding:
        # FWD = blue
        # dX  = green
        # dK  = orange
        #
        # No hatch is used because method identity is already explicit
        # on the x-axis.
        # --------------------------------------------------------------
        for phase, column in PHASE_COLUMNS:
            phase_values = np.asarray(
                [
                    (
                        float(rows.loc[method, column])
                        if method in rows.index
                        else 0.0
                    )
                    for method in EXTERNAL_METHOD_ORDER
                ],
                dtype=float,
            )

            for index, method in enumerate(
                EXTERNAL_METHOD_ORDER
            ):
                if not valid[index]:
                    continue

                ax.bar(
                    x[index],
                    phase_values[index],
                    width=0.66,
                    bottom=bottom[index],
                    color=PHASE_COLORS[phase],
                    edgecolor=EDGE_COLOR,
                    linewidth=0.4,
                    zorder=3,
                )

            bottom += phase_values

        # --------------------------------------------------------------
        # Aggregate latency and between-trial standard deviation.
        # --------------------------------------------------------------
        for index, method in enumerate(
            EXTERNAL_METHOD_ORDER
        ):
            if not valid[index]:
                continue

            totals[index] = float(
                rows.loc[
                    method,
                    "total_mean_ms",
                ]
            )

            error_value = rows.loc[
                method,
                "total_between_trial_std_ms",
            ]

            errors[index] = (
                float(error_value)
                if pd.notna(error_value)
                else 0.0
            )

        ax.errorbar(
            x[valid],
            totals[valid],
            yerr=errors[valid],
            fmt="none",
            ecolor=EDGE_COLOR,
            elinewidth=0.65,
            capsize=1.7,
            capthick=0.65,
            zorder=5,
        )

        # --------------------------------------------------------------
        # Triton is unavailable on V100.
        # --------------------------------------------------------------
        triton_index = EXTERNAL_METHOD_ORDER.index(
            "Triton"
        )

        if not valid[triton_index]:
            ax.text(
                x[triton_index],
                ymax * 0.045,
                "N/A",
                ha="center",
                va="bottom",
                fontsize=6.8,
                color="#737A80",
                fontstyle="italic",
            )

        # --------------------------------------------------------------
        # X-axis labels.
        # --------------------------------------------------------------
        panel_tick_labels = (
            base_tick_labels.copy()
        )

        if gpu == "V100":
            panel_tick_labels[2] = "TVM†"

        ax.set_xticks(
            x,
            panel_tick_labels,
            rotation=24,
            ha="right",
        )

        ax.tick_params(
            axis="x",
            length=0,
            pad=2,
        )

        # --------------------------------------------------------------
        # Panel title:
        # keep panel identifier and GPU name together.
        # --------------------------------------------------------------
        panel = chr(
            ord("a") + panel_index
        )

        ax.set_title(
            f"({panel}) {gpu}",
            loc="left",
            pad=3,
            fontsize=8.0,
            fontweight="bold",
        )

        ax.set_ylim(
            0.0,
            ymax * 1.05,
        )

        style_axis(ax)

    # Shared y-axis label.
    axes[0].set_ylabel(
        "Latency (ms)"
    )

    # ================================================================
    # Shared phase legend.
    # ================================================================
    phase_handles = [
        Patch(
            facecolor=PHASE_COLORS[phase],
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
            label=phase,
        )
        for phase, _ in PHASE_COLUMNS
    ]

    fig.legend(
        handles=phase_handles,
        frameon=False,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        columnspacing=1.0,
        handletextpad=0.4,
        handlelength=1.4,
        borderaxespad=0.0,
    )

    # ================================================================
    # Final layout.
    # ================================================================
    fig.tight_layout(
        rect=(0.01, 0.02, 0.99, 0.90),
        w_pad=0.8,
    )

    return save_figure(
        fig,
        figures_dir / "fig_external_baselines",
    )

def prepare_shape_data(root: Path) -> pd.DataFrame:
    a6000 = read_csv(
        root,
        "results/shape_sweep_ampere/a6000_full/aggregate_comparison.csv",
    )
    require_columns(
        a6000,
        ["config_id", "sweep_group", "phase", "warp_speedup_vs_pytorch"],
        "A6000 shape sweep",
    )
    a6000 = a6000[a6000["phase"].str.lower() == "total"].copy()
    a6000["gpu_short"] = "RTX A6000"
    v100 = read_csv(
        root,
        "results/shape_sweep_volta/v100_full_w16_b256/aggregate_total_comparison.csv",
    )
    require_columns(
        v100,
        ["config_id", "sweep_group", "warp_speedup_vs_pytorch"],
        "V100 shape sweep",
    )
    v100 = v100.copy()
    v100["gpu_short"] = "V100"
    combined = pd.concat([a6000, v100], ignore_index=True, sort=False)
    if combined.duplicated(["gpu_short", "config_id"]).any():
        raise ValueError("Shape sweep contains duplicate GPU/config rows")
    if (combined["warp_speedup_vs_pytorch"] <= 0).any():
        raise ValueError("Shape sweep contains a non-positive speedup")
    return combined



def shape_generalization_figure_detailed(
    shape_data: pd.DataFrame, figures_dir: Path
) -> tuple[list[Path], pd.DataFrame]:
    """Show warp-tiled kernel speedup across workload-shape groups."""
    gpu_names = ["RTX A6000", "V100"]
    all_speedups = shape_data["warp_speedup_vs_pytorch"].to_numpy(dtype=float)
    ymin = min(0.9, float(all_speedups.min()) * 0.94)
    ymax = float(all_speedups.max()) * 1.10
    fig, axes = plt.subplots(
        1, 2, figsize=(ACM_TEXT_WIDTH_IN, 2.55), sharey=True
    )
    summary_rows: list[dict[str, float | int | str]] = []
    for ax, gpu_name in zip(axes, gpu_names, strict=True):
        gpu_data = shape_data[shape_data["gpu_short"] == gpu_name]
        groups = ordered_groups(gpu_data["sweep_group"])
        for group_index, group in enumerate(groups):
            values = gpu_data[gpu_data["sweep_group"] == group][
                "warp_speedup_vs_pytorch"
            ].to_numpy(dtype=float)
            jitter = np.linspace(-0.16, 0.16, len(values)) if len(values) > 1 else [0.0]
            ax.vlines(
                group_index,
                float(values.min()),
                float(values.max()),
                color="#8F969B",
                linewidth=1.0,
                zorder=1,
            )
            ax.scatter(
                group_index + np.asarray(jitter),
                values,
                s=24,
                color=GPU_COLORS[gpu_name],
                edgecolor=EDGE_COLOR,
                linewidth=0.35,
                alpha=0.88,
                label="Shape" if group_index == 0 else None,
                zorder=3,
            )
            ax.scatter(
                group_index,
                geometric_mean(values),
                marker="D",
                s=28,
                color="#263238",
                edgecolor="white",
                linewidth=0.4,
                label="Group geomean" if group_index == 0 else None,
                zorder=4,
            )
        speedups = gpu_data["warp_speedup_vs_pytorch"].to_numpy(dtype=float)
        gmean = geometric_mean(speedups)
        summary_rows.append(
            {
                "gpu": gpu_name,
                "shapes": len(speedups),
                "geomean_speedup": gmean,
                "min_speedup": float(speedups.min()),
                "max_speedup": float(speedups.max()),
                "mean_runtime_reduction_pct": float(
                    (1.0 - 1.0 / speedups).mean() * 100.0
                ),
            }
        )
        ax.axhline(1.0, color="#737A80", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(groups)), [group.title() for group in groups], rotation=18)
        panel = "(a)" if gpu_name == gpu_names[0] else "(b)"
        ax.set_title(f"{panel} {gpu_name}", loc="left")
        ax.set_ylim(ymin, ymax)
        style_axis(ax)
    axes[0].set_ylabel("Warp-tiled speedup vs PyTorch (×)")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    paths = save_figure(fig, figures_dir / "fig_shape_generalization")
    return paths, pd.DataFrame(summary_rows)


def holdout_figure_detailed(
    holdout: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Evaluate the frozen PathTile policy on held-out shapes."""

    # ------------------------------------------------------------------
    # Validate input data.
    # ------------------------------------------------------------------
    require_columns(
        holdout,
        [
            "config_id",
            "gpu",
            "frozen_regret_pct",
            "exact_oracle_schedule",
        ],
        "Three-GPU holdout",
    )

    gpu_order = [
        "A100",
        "A6000",
        "V100",
    ]

    gpu_display = {
        "A100": "A100",
        "A6000": "RTX A6000",
        "V100": "V100",
    }

    counts = (
        holdout
        .groupby("gpu")
        .size()
        .reindex(gpu_order)
    )

    if len(holdout) != 30 or not (counts == 10).all():
        raise ValueError(
            f"Expected 10 holdout shapes per GPU, "
            f"got {counts.to_dict()}"
        )

    holdout = holdout.copy()

    # Small negative values can arise from measurement/numerical noise.
    # Regret is displayed from zero upward.
    holdout["plot_regret_pct"] = (
        holdout["frozen_regret_pct"]
        .clip(lower=0.0)
    )

    # ------------------------------------------------------------------
    # Figure layout.
    # ------------------------------------------------------------------
    fig, (ax, ax_cdf) = plt.subplots(
        1,
        2,
        figsize=(ACM_TEXT_WIDTH_IN, 2.50),
        gridspec_kw={
            "width_ratios": [1.0, 1.05],
            "wspace": 0.30,
        },
    )

    # ================================================================
    # (a) Per-GPU holdout regret
    # ================================================================
    for index, gpu in enumerate(gpu_order):

        values = (
            holdout.loc[
                holdout["gpu"] == gpu,
                "plot_regret_pct",
            ]
            .to_numpy(dtype=float)
        )

        # Deterministic horizontal spacing makes overlapping points
        # visible without implying additional information.
        jitter = np.linspace(
            -0.16,
            0.16,
            len(values),
        )

        # Individual held-out shapes.
        ax.scatter(
            index + jitter,
            values,
            s=24,
            color=GPU_COLORS[gpu],
            edgecolor=EDGE_COLOR,
            linewidth=0.35,
            alpha=0.90,
            label="Held-out shape"
            if index == 0
            else None,
            zorder=3,
        )

        # Per-GPU arithmetic mean regret.
        mean_value = float(
            values.mean()
        )

        ax.scatter(
            index,
            mean_value,
            marker="D",
            s=34,
            color="#263238",
            edgecolor=EDGE_COLOR,
            linewidth=0.35,
            label="Mean"
            if index == 0
            else None,
            zorder=4,
        )

    # --------------------------------------------------------------
    # Regret thresholds.
    # These are explained in the caption rather than added to
    # the legend to keep the panel visually compact.
    # --------------------------------------------------------------
    ax.axhline(
        1.0,
        color="#737A80",
        linestyle="--",
        linewidth=0.8,
        zorder=1,
    )

    ax.axhline(
        2.0,
        color="#9AA0A6",
        linestyle=":",
        linewidth=0.8,
        zorder=1,
    )

    ax.set_xticks(
        range(len(gpu_order)),
        [
            gpu_display[gpu]
            for gpu in gpu_order
        ],
    )

    ax.set_ylabel(
        "Frozen-configuration regret (%)"
    )

    ax.set_ylim(
        0.0,
        max(
            2.25,
            float(
                holdout["plot_regret_pct"].max()
            ) * 1.12,
        ),
    )

    ax.set_title(
        "(a) Per-GPU holdout regret",
        loc="left",
        pad=3,
        fontsize=8.0,
        fontweight="bold",
    )

    style_axis(ax)

    # Legend contains only actual data encodings.
    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        columnspacing=0.9,
        handletextpad=0.35,
        borderaxespad=0.2,
    )

    # ================================================================
    # (b) Empirical CDF
    # ================================================================
    cdf_line_styles = {
        "A100": "-",
        "A6000": "--",
        "V100": "-.",
        "All": "-",
    }

    for gpu in [*gpu_order, "All"]:

        if gpu == "All":
            values = np.sort(
                holdout[
                    "plot_regret_pct"
                ].to_numpy(dtype=float)
            )

            color = GPU_COLORS["All"]
            linewidth = 1.8
            label = "All"

        else:
            values = np.sort(
                holdout.loc[
                    holdout["gpu"] == gpu,
                    "plot_regret_pct",
                ].to_numpy(dtype=float)
            )

            color = GPU_COLORS[gpu]
            linewidth = 1.0
            label = gpu_display[gpu]

        cumulative = (
            np.arange(
                1,
                len(values) + 1,
                dtype=float,
            )
            / len(values)
            * 100.0
        )

        ax_cdf.step(
            np.concatenate(
                ([0.0], values)
            ),
            np.concatenate(
                ([0.0], cumulative)
            ),
            where="post",
            color=color,
            linewidth=linewidth,
            linestyle=cdf_line_styles[gpu],
            label=label,
        )

    # --------------------------------------------------------------
    # Same 1% and 2% regret thresholds as in panel (a).
    #
    # Exact 26/30 and 29/30 statistics are reported in the caption
    # and main text rather than annotated inside the plot.
    # --------------------------------------------------------------
    ax_cdf.axvline(
        1.0,
        color="#8F969B",
        linestyle="--",
        linewidth=0.8,
        zorder=1,
    )

    ax_cdf.axvline(
        2.0,
        color="#B8BDC1",
        linestyle=":",
        linewidth=0.8,
        zorder=1,
    )

    ax_cdf.set_xlim(
        -0.03,
        2.15,
    )

    ax_cdf.set_ylim(
        0.0,
        103.0,
    )

    ax_cdf.set_xlabel(
        "Frozen-configuration regret (%)"
    )

    ax_cdf.set_ylabel(
        "Shapes at or below threshold (%)"
    )

    ax_cdf.set_title(
        "(b) Empirical CDF",
        loc="left",
        pad=3,
        fontsize=8.0,
        fontweight="bold",
    )

    ax_cdf.legend(
        frameon=False,
        ncol=2,
        loc="lower right",
        columnspacing=0.9,
        handlelength=1.8,
    )

    style_axis(
        ax_cdf,
        grid_axis="both",
    )

    # ------------------------------------------------------------------
    # Final layout.
    # ------------------------------------------------------------------
    fig.tight_layout(
        rect=(0.01, 0.02, 0.99, 0.98),
        w_pad=1.0,
    )

    return save_figure(
        fig,
        figures_dir / "fig_holdout_regret",
    )

def collect_gpu_utilization(root: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    serial_root = root / "results/end_to_end/a6000_final_serial"
    for trial in [1, 2, 3]:
        for backend in BACKEND_ORDER:
            path = serial_root / f"trial{trial}/{backend}/training_epoch_system_summary.csv"
            if not path.is_file():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            require_columns(frame, ["epoch", "util_gpu_avg_pct"], str(path))
            steady = frame[frame["epoch"].isin([2, 3, 4, 5])]
            if sorted(steady["epoch"].astype(int).tolist()) != [2, 3, 4, 5]:
                raise ValueError(f"Invalid steady-state epochs in {path}")
            rows.append(
                {
                    "trial": trial,
                    "backend": backend,
                    "gpu_util_avg_pct": float(steady["util_gpu_avg_pct"].mean()),
                }
            )
    per_trial = pd.DataFrame(rows)
    return (
        per_trial.groupby("backend", as_index=False)["gpu_util_avg_pct"]
        .agg(["mean", "std"])
        .reset_index()
        .rename(columns={"mean": "gpu_util_mean_pct", "std": "gpu_util_std_pct"})
    )



def end_to_end_figure_and_table_detailed(
    root: Path,
    figures_dir: Path,
) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    """Render the detailed end-to-end PathTile training evaluation."""

    # ================================================================
    # Load canonical end-to-end results
    # ================================================================
    backend = read_csv(
        root,
        "results/end_to_end/a6000_final_serial/"
        "final_backend_aggregate.csv",
    )

    paper = read_csv(
        root,
        "results/end_to_end/a6000_final_serial/"
        "final_paper_summary.csv",
    )

    paired = read_csv(
        root,
        "results/end_to_end/a6000_final_serial/"
        "final_paired_speedups.csv",
    )

    # ================================================================
    # Validate input schemas
    # ================================================================
    require_columns(
        backend,
        [
            "backend",
            "trials",
            "epoch_wall_mean_s_mean",
            "epoch_wall_mean_s_between_trial_std",
            "throughput_samples_s_mean",
            "model_fwd_mean_ms_mean",
            "backward_mean_ms_mean",
            "gpu_stages_mean_ms_mean",
            "gpu_stages_mean_ms_between_trial_std",
            "peak_memory_gib_mean",
            "test_rmsle_mean",
        ],
        "Serial end-to-end backend aggregate",
    )

    require_columns(
        paper,
        [
            "epoch_speedup_geomean",
            "model_fwd_speedup_geomean",
            "backward_speedup_geomean",
            "gpu_stages_speedup_geomean",
            "max_rmsle_abs_difference",
        ],
        "Serial end-to-end paper summary",
    )

    if len(paired) != 3:
        raise ValueError(
            f"Expected three paired end-to-end trials, got {len(paired)}"
        )

    max_rmsle_difference = float(
        paper["max_rmsle_abs_difference"].iloc[0]
    )

    if max_rmsle_difference > 1e-12:
        raise ValueError(
            "End-to-end correctness validation failed: "
            f"maximum RMSLE difference is {max_rmsle_difference:.3e}"
        )

    expected_backends = set(BACKEND_ORDER)
    actual_backends = set(backend["backend"])

    if actual_backends != expected_backends:
        raise ValueError(
            "Unexpected end-to-end backends: "
            f"expected {sorted(expected_backends)}, "
            f"got {sorted(actual_backends)}"
        )

    if backend["backend"].duplicated().any():
        raise ValueError(
            "End-to-end aggregate contains duplicate backend rows"
        )

    # ================================================================
    # Add steady-state GPU-utilization statistics
    # ================================================================
    utilization = collect_gpu_utilization(root)

    merged = backend.merge(
        utilization,
        on="backend",
        validate="one_to_one",
    )

    indexed = (
        merged
        .set_index("backend")
        .loc[BACKEND_ORDER]
        .copy()
    )

    # The internal backend name is "warp", but this end-to-end
    # experiment evaluates the final PathTile-selected configuration.
    method_labels = [
        SYSTEM_BACKEND_LABELS[backend_name]
        for backend_name in BACKEND_ORDER
    ]

    # Restrained method comparison for panel (a): PyTorch is neutral and
    # PathTile receives the single method-accent color. This prevents backend
    # identity from reusing the phase colors employed in panels (b) and (c).
    wall_method_colors = [
        "#D8DBDE",
        COLORS["PathTile"],
    ]

    method_x = np.arange(
        len(BACKEND_ORDER),
        dtype=float,
    )

    # ================================================================
    # Prepare measurements
    # ================================================================
    wall_values = indexed[
        "epoch_wall_mean_s_mean"
    ].to_numpy(dtype=float)

    wall_errors = (
        indexed["epoch_wall_mean_s_between_trial_std"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    model_fwd_values = indexed[
        "model_fwd_mean_ms_mean"
    ].to_numpy(dtype=float)

    backward_values = indexed[
        "backward_mean_ms_mean"
    ].to_numpy(dtype=float)

    gpu_stage_values = indexed[
        "gpu_stages_mean_ms_mean"
    ].to_numpy(dtype=float)

    gpu_stage_errors = (
        indexed["gpu_stages_mean_ms_between_trial_std"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    other_gpu_values = (
        gpu_stage_values
        - model_fwd_values
        - backward_values
    )

    if np.any(other_gpu_values < -1e-6):
        raise ValueError(
            "Recorded Model FWD and Backward components exceed "
            "the measured total GPU-stage time"
        )

    # Clamp only negligible negative values caused by floating-point
    # subtraction.
    other_gpu_values = np.maximum(
        other_gpu_values,
        0.0,
    )

    # Store the derived component for the returned paper table.
    indexed["other_gpu_mean_ms"] = other_gpu_values

    # ================================================================
    # Figure layout
    # ================================================================
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(ACM_TEXT_WIDTH_IN, 2.42),
        gridspec_kw={
            "width_ratios": [0.82, 1.00, 1.18],
        },
    )

    ax_wall, ax_gpu, ax_scope = axes

    # ================================================================
    # (a) Epoch wall time
    # ================================================================
    wall_bars = ax_wall.bar(
        method_x,
        wall_values,
        yerr=wall_errors,
        capsize=1.7,
        width=0.62,
        color=wall_method_colors,
        edgecolor=EDGE_COLOR,
        linewidth=0.4,
        zorder=3,
    )

    # Only the two aggregate values are shown in this panel.
    for bar, value in zip(
        wall_bars,
        wall_values,
        strict=True,
    ):
        ax_wall.text(
            bar.get_x() + bar.get_width() / 2.0,
            value + wall_values.max() * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )

    ax_wall.set_xticks(
        method_x,
        method_labels,
    )

    ax_wall.set_ylabel(
        "Epoch wall time (s)"
    )

    ax_wall.set_ylim(
        0.0,
        wall_values.max() * 1.14,
    )

    ax_wall.set_title(
        "(a) Epoch wall time",
        loc="left",
        pad=3,
        fontsize=8.0,
        fontweight="bold",
    )

    style_axis(
        ax_wall,
        grid_axis="y",
    )

    # ================================================================
    # (b) Recorded GPU stages
    # ================================================================
    # Solid semantic phase colors; no hatch patterns.
    gpu_components = [
        (
            "Model FWD",
            model_fwd_values,
            E2E_COLORS["Model FWD"],
        ),
        (
            "Backward",
            backward_values,
            E2E_COLORS["Backward"],
        ),
        (
            "Other GPU",
            other_gpu_values,
            E2E_COLORS["Other GPU"],
        ),
    ]

    gpu_bottom = np.zeros(
        len(BACKEND_ORDER),
        dtype=float,
    )

    component_handles: list[Patch] = []

    for (
        label,
        values,
        color,
    ) in gpu_components:
        ax_gpu.bar(
            method_x,
            values,
            width=0.62,
            bottom=gpu_bottom,
            color=color,
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
            zorder=3,
        )

        gpu_bottom += values

        component_handles.append(
            Patch(
                facecolor=color,
                edgecolor=EDGE_COLOR,
                linewidth=0.4,
                label=label,
            )
        )

    ax_gpu.errorbar(
        method_x,
        gpu_stage_values,
        yerr=gpu_stage_errors,
        fmt="none",
        ecolor=EDGE_COLOR,
        elinewidth=0.65,
        capsize=1.7,
        capthick=0.65,
        zorder=5,
    )

    # Only aggregate GPU-stage time is shown above each stack.
    for xpos, value in zip(
        method_x,
        gpu_stage_values,
        strict=True,
    ):
        ax_gpu.text(
            xpos,
            value + gpu_stage_values.max() * 0.025,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )

    ax_gpu.set_xticks(
        method_x,
        method_labels,
    )

    ax_gpu.set_ylabel(
        "Time per batch (ms)"
    )

    ax_gpu.set_ylim(
        0.0,
        gpu_stage_values.max() * 1.14,
    )

    ax_gpu.set_title(
        "(b) Recorded GPU stages",
        loc="left",
        pad=3,
        fontsize=8.0,
        fontweight="bold",
    )

    style_axis(
        ax_gpu,
        grid_axis="y",
    )

    # ================================================================
    # (c) Speedup by scope
    # ================================================================
    scope_labels = [
        "Model FWD",
        "Backward",
        "GPU stages",
        "Epoch wall",
    ]

    scope_values = np.asarray(
        [
            float(
                paper[
                    "model_fwd_speedup_geomean"
                ].iloc[0]
            ),
            float(
                paper[
                    "backward_speedup_geomean"
                ].iloc[0]
            ),
            float(
                paper[
                    "gpu_stages_speedup_geomean"
                ].iloc[0]
            ),
            float(
                paper[
                    "epoch_speedup_geomean"
                ].iloc[0]
            ),
        ],
        dtype=float,
    )

    if (
        not np.isfinite(scope_values).all()
        or np.any(scope_values <= 0.0)
    ):
        raise ValueError(
            f"Invalid end-to-end speedup values: {scope_values}"
        )

    scope_y = np.arange(
        len(scope_labels),
        dtype=float,
    )

    # Semantic end-to-end palette:
    # Model FWD = muted blue
    # Backward  = muted purple (aggregate backward stage, not dK)
    # GPU stages = muted teal
    # Epoch wall = neutral gray
    scope_colors = [
        E2E_COLORS["Model FWD"],
        E2E_COLORS["Backward"],
        E2E_COLORS["GPU stages"],
        E2E_COLORS["Epoch wall"],
    ]

    # Bars start at the 1× PyTorch baseline.
    ax_scope.barh(
        scope_y,
        scope_values - 1.0,
        left=1.0,
        height=0.56,
        color=scope_colors,
        edgecolor=EDGE_COLOR,
        linewidth=0.4,
        zorder=3,
    )

    # Speedup values are the primary quantity in this panel,
    # so retain direct annotations.
    for ypos, value in zip(
        scope_y,
        scope_values,
        strict=True,
    ):
        ax_scope.text(
            value + 0.018,
            ypos,
            f"{value:.2f}×",
            ha="left",
            va="center",
            fontsize=7.0,
        )

    # 1× denotes equal performance with PyTorch.
    ax_scope.axvline(
        1.0,
        color="#737A80",
        linestyle=":",
        linewidth=0.8,
        zorder=1,
    )

    ax_scope.set_yticks(
        scope_y,
        scope_labels,
    )

    ax_scope.invert_yaxis()

    ax_scope.set_xlim(
        0.98,
        max(
            2.08,
            float(scope_values.max()) + 0.14,
        ),
    )

    ax_scope.set_xlabel(
        "Speedup vs PyTorch (×)"
    )

    ax_scope.set_title(
        "(c) Speedup by scope",
        loc="left",
        pad=3,
        fontsize=8.0,
        fontweight="bold",
    )

    style_axis(
        ax_scope,
        grid_axis="x",
    )

    # ================================================================
    # Shared legend
    # ================================================================
    # The shared legend describes the stacked GPU-stage components
    # in panel (b). Panel (c) is directly labeled on the y-axis.
    ax_gpu.legend(
        handles=component_handles,
        frameon=False,
        ncol=3,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.12),
        columnspacing=0.8,
        handletextpad=0.35,
        handlelength=1.2,
        borderaxespad=0.0,
    )

    # ================================================================
    # Final layout and export
    # ================================================================
    # Use explicit spacing instead of tight_layout() because the legend
    # belongs to the figure rather than to an individual axes.
    fig.subplots_adjust(
        left=0.075,
        right=0.985,
        bottom=0.20,
        top=0.82,
        wspace=0.52,
    )

    paths = save_figure(
        fig,
        figures_dir / "fig_end_to_end_a6000",
    )

    # ================================================================
    # Paper-facing summary table
    # ================================================================
    display = (
        indexed
        .reset_index()[
            [
                "backend",
                "trials",
                "epoch_wall_mean_s_mean",
                "epoch_wall_mean_s_between_trial_std",
                "throughput_samples_s_mean",
                "model_fwd_mean_ms_mean",
                "backward_mean_ms_mean",
                "other_gpu_mean_ms",
                "gpu_stages_mean_ms_mean",
                "gpu_stages_mean_ms_between_trial_std",
                "gpu_util_mean_pct",
                "gpu_util_std_pct",
                "peak_memory_gib_mean",
                "test_rmsle_mean",
            ]
        ]
        .copy()
    )

    display["backend"] = display["backend"].map(
        SYSTEM_BACKEND_LABELS
    )

    display.columns = [
        "Backend",
        "Trials",
        "Epoch wall mean (s)",
        "Epoch wall std (s)",
        "Throughput (samples/s)",
        "Model FWD (ms/batch)",
        "Backward (ms/batch)",
        "Other GPU (ms/batch)",
        "All GPU stages (ms/batch)",
        "All GPU stages std (ms/batch)",
        "GPU utilization mean (%)",
        "GPU utilization std (%)",
        "Peak memory (GiB)",
        "Test RMSLE",
    ]

    return paths, display, paper

# ---------------------------------------------------------------------------
# Optional compact figures
# ---------------------------------------------------------------------------
# These single-message variants are kept for talks or space-constrained drafts.
# The default main-paper output uses the interpretive multi-panel figures above,
# with numeric annotations intentionally removed.


def cross_gpu_figure(cross: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Compare total PyTorch and warp-tiled latency across GPUs."""
    selected = cross[cross["method"].isin(["PyTorch", "Warp"])].copy()
    indexed = selected.set_index(["gpu", "method"])

    x = np.arange(len(GPU_ORDER), dtype=float)
    width = 0.34
    fig, ax = plt.subplots(figsize=(ACM_COLUMN_WIDTH_IN, 2.25))

    specs = [
        ("PyTorch", -width / 2, COLORS["PyTorch"], "PyTorch"),
        ("Warp", width / 2, COLORS["Warp"], KERNEL_METHOD_LABELS["Warp"]),
    ]
    ymax = 0.0
    for method, offset, color, label in specs:
        rows = indexed.loc[(GPU_ORDER, method), :].reset_index(level="method")
        values = rows["total_mean_ms"].to_numpy(dtype=float)
        errors = rows["total_between_trial_std_ms"].fillna(0.0).to_numpy(dtype=float)
        ymax = max(ymax, float((values + errors).max()))
        ax.bar(
            x + offset,
            values,
            width,
            yerr=errors,
            capsize=1.7,
            color=color,
            edgecolor=EDGE_COLOR,
            linewidth=0.45,
            label=label,
            zorder=3,
        )

    warp_tiled_rows = indexed.loc[(GPU_ORDER, "Warp"), :]
    for xpos, (_, row) in zip(
        x + width / 2, warp_tiled_rows.iterrows(), strict=True
    ):
        value = float(row["total_mean_ms"])
        speedup = float(row["total_speedup_vs_pytorch"])
        ax.text(
            xpos,
            value + ymax * 0.045,
            f"{speedup:.2f}×",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
        )

    ax.set_xticks(x, [GPU_SHORT[gpu] for gpu in GPU_ORDER])
    ax.set_ylabel("Total latency (ms)")
    ax.set_ylim(0, ymax * 1.18)
    ax.legend(frameon=False, ncol=2, loc="upper right", columnspacing=0.9)
    style_axis(ax)
    fig.tight_layout(pad=0.25)
    return save_figure(fig, figures_dir / "fig_cross_gpu_latency")

def v100_ablation_figure(
    cross: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Show cumulative optimization progression on V100."""

    data = (
        cross[cross["gpu"] == GPU_ORDER[2]]
        .set_index("method")
        .loc[METHOD_ORDER]
    )

    x = np.arange(len(METHOD_ORDER), dtype=float)

    fig, ax = plt.subplots(
        figsize=(ACM_COLUMN_WIDTH_IN, 2.35)
    )

    bottom = np.zeros(len(METHOD_ORDER), dtype=float)

    # Same phase colors used throughout the paper:
    # FWD = blue, dX = green, dK = orange.
    for phase, column in PHASE_COLUMNS:
        phase_values = data[column].to_numpy(dtype=float)

        ax.bar(
            x,
            phase_values,
            bottom=bottom,
            width=0.68,
            color=PHASE_COLORS[phase],
            edgecolor=EDGE_COLOR,
            linewidth=0.4,
            label=phase,
            zorder=3,
        )

        bottom += phase_values

    # Between-trial standard deviation of aggregate latency.
    errors = (
        data["total_between_trial_std_ms"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    ax.errorbar(
        x,
        bottom,
        yerr=errors,
        fmt="none",
        ecolor=EDGE_COLOR,
        elinewidth=0.65,
        capsize=1.8,
        capthick=0.65,
        zorder=5,
    )

    # Report only the final warp-tiled speedup.
    final_speedup = float(
        data.loc["Warp", "total_speedup_vs_pytorch"]
    )

    ax.text(
        x[-1],
        bottom[-1] + bottom.max() * 0.035,
        f"{final_speedup:.2f}×",
        ha="center",
        va="bottom",
        fontsize=7.0,
        fontweight="bold",
    )

    display_methods = [KERNEL_METHOD_LABELS[method] for method in METHOD_ORDER]

    ax.set_xticks(
        x,
        display_methods,
        rotation=18,
        ha="right",
    )

    ax.set_ylabel("Latency (ms)")
    ax.set_ylim(0.0, bottom.max() * 1.12)

    style_axis(ax)

    # Phase legend only.
    ax.legend(
        frameon=False,
        ncol=3,
        loc="upper right",
        columnspacing=0.8,
        handletextpad=0.35,
    )

    fig.tight_layout(pad=0.25)

    return save_figure(
        fig,
        figures_dir / "fig_v100_ablation",
    )

def external_baseline_figure(
    external: pd.DataFrame,
    external_paper: pd.DataFrame,
    figures_dir: Path,
) -> list[Path]:
    """Compare final PathTile latency with external compiler baselines."""
    require_columns(
        external,
        ["gpu", "method", "total_mean_ms", "total_between_trial_std_ms"],
        "External compiler baselines",
    )
    gpu_order = ["A100", "RTX A6000", "V100"]
    paper = external_paper.set_index("gpu").loc[gpu_order]
    method_colors = {
        "PyTorch": COLORS["PyTorch"],
        "Triton": COLORS["Triton"],
        "TVM-TIRx": COLORS["TVM-TIRx"],
        "PathTile": COLORS["PathTile"],
    }
    labels = ["PyT", "Triton", "TVM", "PT"]
    x = np.arange(len(EXTERNAL_METHOD_ORDER), dtype=float)
    ymax = float(external["total_mean_ms"].max())
    fig, axes = plt.subplots(
        1, 3, figsize=(ACM_TEXT_WIDTH_IN, 2.2), sharey=True
    )

    for panel_index, (ax, gpu) in enumerate(zip(axes, gpu_order, strict=True)):
        rows = external[external["gpu"] == gpu].set_index("method")
        for index, method in enumerate(EXTERNAL_METHOD_ORDER):
            if method not in rows.index:
                ax.text(index, ymax * 0.025, "—", ha="center", va="bottom")
                continue
            value = float(rows.loc[method, "total_mean_ms"])
            error_value = rows.loc[method, "total_between_trial_std_ms"]
            error = float(error_value) if pd.notna(error_value) else 0.0
            ax.bar(
                index,
                value,
                width=0.68,
                yerr=error,
                capsize=1.5,
                color=method_colors[method],
                edgecolor=EDGE_COLOR,
                linewidth=0.6 if method == "PathTile" else 0.35,
                zorder=3,
            )

        speedup = float(paper.loc[gpu, "pathtile_speedup_vs_best_compiler"])
        pathtile_value = float(rows.loc["PathTile", "total_mean_ms"])
        ax.text(
            x[-1],
            pathtile_value + ymax * 0.04,
            f"{speedup:.2f}×",
            ha="center",
            va="bottom",
            fontsize=7.0,
            fontweight="bold",
        )
        ax.set_xticks(x, labels)
        ax.set_title(f"({chr(97 + panel_index)}) {gpu}", loc="left")
        ax.set_ylim(0, ymax * 1.13)
        style_axis(ax)

    axes[0].set_ylabel("Total latency (ms)")
    fig.tight_layout(pad=0.35, w_pad=0.7)
    return save_figure(fig, figures_dir / "fig_external_baselines")


def shape_generalization_figure(
    shape_data: pd.DataFrame, figures_dir: Path
) -> tuple[list[Path], pd.DataFrame]:
    """Show warp-tiled speedups and group geometric means by shape."""
    gpu_names = ["RTX A6000", "V100"]
    all_speedups = shape_data["warp_speedup_vs_pytorch"].to_numpy(dtype=float)
    ymin = min(0.9, float(all_speedups.min()) * 0.94)
    ymax = float(all_speedups.max()) * 1.08
    fig, axes = plt.subplots(
        1, 2, figsize=(ACM_TEXT_WIDTH_IN, 2.35), sharey=True
    )
    summary_rows: list[dict[str, float | int | str]] = []

    for panel_index, (ax, gpu_name) in enumerate(
        zip(axes, gpu_names, strict=True)
    ):
        gpu_data = shape_data[shape_data["gpu_short"] == gpu_name]
        groups = ordered_groups(gpu_data["sweep_group"])
        for group_index, group in enumerate(groups):
            values = gpu_data.loc[
                gpu_data["sweep_group"] == group, "warp_speedup_vs_pytorch"
            ].to_numpy(dtype=float)
            jitter = (
                np.linspace(-0.13, 0.13, len(values))
                if len(values) > 1
                else np.asarray([0.0])
            )
            ax.scatter(
                group_index + jitter,
                values,
                s=18,
                color=GPU_COLORS[gpu_name],
                edgecolor=EDGE_COLOR,
                linewidth=0.3,
                alpha=0.85,
                zorder=3,
            )
            ax.scatter(
                group_index,
                geometric_mean(values),
                marker="D",
                s=25,
                color="#263238",
                edgecolor="white",
                linewidth=0.35,
                zorder=4,
            )

        speedups = gpu_data["warp_speedup_vs_pytorch"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "gpu": gpu_name,
                "shapes": len(speedups),
                "geomean_speedup": geometric_mean(speedups),
                "min_speedup": float(speedups.min()),
                "max_speedup": float(speedups.max()),
                "mean_runtime_reduction_pct": float(
                    (1.0 - 1.0 / speedups).mean() * 100.0
                ),
            }
        )
        ax.axhline(1.0, color="#737A80", linestyle="--", linewidth=0.75)
        ax.set_xticks(range(len(groups)), [group.title() for group in groups])
        ax.set_title(f"({chr(97 + panel_index)}) {gpu_name}", loc="left")
        ax.set_ylim(ymin, ymax)
        style_axis(ax)

    axes[0].set_ylabel("Warp-tiled speedup vs PyTorch (×)")
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#8F969B",
            markeredgecolor=EDGE_COLOR,
            markeredgewidth=0.3,
            markersize=4.5,
            label="Shape",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=EDGE_COLOR,
            markeredgecolor="white",
            markersize=4.5,
            label="Geometric mean",
        ),
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.0,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=0.35, w_pad=0.8)
    paths = save_figure(fig, figures_dir / "fig_shape_generalization")
    return paths, pd.DataFrame(summary_rows)


def holdout_figure(holdout: pd.DataFrame, figures_dir: Path) -> list[Path]:
    """Use a single empirical CDF for the frozen PathTile policy."""
    require_columns(
        holdout,
        ["config_id", "gpu", "frozen_regret_pct", "exact_oracle_schedule"],
        "Three-GPU holdout",
    )
    counts = holdout.groupby("gpu").size().reindex(["A100", "A6000", "V100"])
    if len(holdout) != 30 or not (counts == 10).all():
        raise ValueError(f"Expected 10 holdout shapes per GPU, got {counts.to_dict()}")

    data = holdout.copy()
    data["plot_regret_pct"] = data["frozen_regret_pct"].clip(lower=0.0)
    styles = {
        "A100": (GPU_COLORS["A100"], "-", 1.0),
        "A6000": (GPU_COLORS["A6000"], "--", 1.0),
        "V100": (GPU_COLORS["V100"], "-.", 1.0),
        "All": (GPU_COLORS["All"], "-", 1.8),
    }
    fig, ax = plt.subplots(figsize=(ACM_COLUMN_WIDTH_IN, 2.35))

    for gpu in ["A100", "A6000", "V100", "All"]:
        if gpu == "All":
            values = np.sort(data["plot_regret_pct"].to_numpy(dtype=float))
        else:
            values = np.sort(
                data.loc[data["gpu"] == gpu, "plot_regret_pct"].to_numpy(
                    dtype=float
                )
            )
        cumulative = np.arange(1, len(values) + 1, dtype=float) / len(values) * 100
        color, linestyle, linewidth = styles[gpu]
        ax.step(
            np.concatenate(([0.0], values)),
            np.concatenate(([0.0], cumulative)),
            where="post",
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            label=gpu,
        )

    for threshold in [1.0, 2.0]:
        ax.axvline(threshold, color="#B8BDC1", linestyle=":", linewidth=0.7)
    ax.set_xlim(0, max(2.05, float(data["plot_regret_pct"].max()) * 1.05))
    ax.set_ylim(0, 102)
    ax.set_xlabel("Frozen PathTile regret (%)")
    ax.set_ylabel("Held-out shapes (%)")
    ax.legend(frameon=False, ncol=2, loc="lower right", columnspacing=0.8)
    style_axis(ax, grid_axis="both")
    fig.tight_layout(pad=0.25)
    return save_figure(fig, figures_dir / "fig_holdout_regret")


def end_to_end_figure_and_table(
    root: Path,
    figures_dir: Path,
) -> tuple[list[Path], pd.DataFrame, pd.DataFrame]:
    """Render a compact PyTorch-versus-PathTile end-to-end comparison."""

    _, display, paper = end_to_end_figure_and_table_detailed(
        root,
        figures_dir,
    )

    # ------------------------------------------------------------------
    # Final end-to-end systems:
    # PyTorch versus the PathTile-configured implementation.
    # ------------------------------------------------------------------
    indexed = (
        display
        .set_index("Backend")
        .loc[["PyTorch", "PathTile"]]
    )

    # ------------------------------------------------------------------
    # Figure layout.
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(ACM_TEXT_WIDTH_IN, 2.25),
        gridspec_kw={
            "width_ratios": [0.80, 1.20],
            "wspace": 0.38,
        },
    )

    # ================================================================
    # (a) End-to-end training epoch wall time
    # ================================================================
    wall_values = indexed[
        "Epoch wall mean (s)"
    ].to_numpy(dtype=float)

    wall_errors = (
        indexed["Epoch wall std (s)"]
        .fillna(0.0)
        .to_numpy(dtype=float)
    )

    bars = axes[0].bar(
        [0, 1],
        wall_values,
        yerr=wall_errors,
        capsize=1.7,
        width=0.62,
        color=[
            "#D8DBDE",
            COLORS["PathTile"],
        ],
        edgecolor=EDGE_COLOR,
        linewidth=0.4,
        zorder=3,
    )

    # Exact mean epoch times are useful here because panel (a)
    # contains only two bars.
    for bar, value in zip(
        bars,
        wall_values,
        strict=True,
    ):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            value + wall_values.max() * 0.025,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=7.0,
        )

    axes[0].set_xticks(
        [0, 1],
        [
            "PyTorch",
            "PathTile",
        ],
    )

    axes[0].set_ylabel(
        "Epoch wall time (s)"
    )

    axes[0].set_ylim(
        0.0,
        wall_values.max() * 1.14,
    )

    axes[0].set_title(
        "(a) End-to-end training",
        loc="left",
        pad=3,
        fontsize=8.0,
        fontweight="bold",
    )

    style_axis(
        axes[0],
        grid_axis="y",
    )

    # ================================================================
    # (b) Speedup at different measurement scopes
    # ================================================================
    scope_labels = [
        "Model FWD",
        "Backward",
        "GPU stages",
        "Epoch wall",
    ]

    scope_values = np.asarray(
        [
            float(
                paper[
                    "model_fwd_speedup_geomean"
                ].iloc[0]
            ),
            float(
                paper[
                    "backward_speedup_geomean"
                ].iloc[0]
            ),
            float(
                paper[
                    "gpu_stages_speedup_geomean"
                ].iloc[0]
            ),
            float(
                paper[
                    "epoch_speedup_geomean"
                ].iloc[0]
            ),
        ],
        dtype=float,
    )

    # Scope-specific colors.
    # These encode measurement scope, not kernel design.
    scope_colors = [
        E2E_COLORS["Model FWD"],
        E2E_COLORS["Backward"],
        E2E_COLORS["GPU stages"],
        E2E_COLORS["Epoch wall"],
    ]

    y = np.arange(
        len(scope_labels),
        dtype=float,
    )

    axes[1].barh(
        y,
        scope_values - 1.0,
        left=1.0,
        height=0.60,
        color=scope_colors,
        edgecolor=EDGE_COLOR,
        linewidth=0.35,
        zorder=3,
    )

    # Speedup values are the central quantity of this panel,
    # so they are shown directly next to the bars.
    for ypos, value in zip(
        y,
        scope_values,
        strict=True,
    ):
        axes[1].text(
            value + 0.018,
            ypos,
            f"{value:.2f}×",
            ha="left",
            va="center",
            fontsize=7.0,
        )

    # 1x = equal performance with PyTorch.
    axes[1].axvline(
        1.0,
        color="#737A80",
        linestyle=":",
        linewidth=0.75,
        zorder=1,
    )

    axes[1].set_yticks(
        y,
        scope_labels,
    )

    axes[1].invert_yaxis()

    axes[1].set_xlim(
        0.98,
        max(
            2.05,
            float(scope_values.max()) + 0.12,
        ),
    )

    axes[1].set_xlabel(
        "Speedup vs PyTorch (×)"
    )

    axes[1].set_title(
        "(b) Speedup by scope",
        loc="left",
        pad=3,
        fontsize=8.0,
        fontweight="bold",
    )

    style_axis(
        axes[1],
        grid_axis="x",
    )

    # ------------------------------------------------------------------
    # Final layout.
    # ------------------------------------------------------------------
    fig.tight_layout(
        rect=(0.01, 0.02, 0.99, 0.98),
        pad=0.35,
        w_pad=1.0,
    )

    paths = save_figure(
        fig,
        figures_dir / "fig_end_to_end_a6000",
    )

    return paths, display, paper


def ncu_table(ncu: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        ncu,
        ["phase", "Section Name", "Metric Name", "Metric Unit", "Metric Value"],
        "A100 Nsight Compute key metrics",
    )
    phases = ["Forward", "Bwd-Input", "Bwd-Weight"]
    specifications = [
        ("Duration (ms)", "Duration", "ms", None),
        ("Compute throughput (%)", "Compute (SM) Throughput", "%", None),
        (
            "Memory throughput (%)",
            "Memory Throughput",
            "%",
            "GPU Speed Of Light Throughput",
        ),
        ("L2 hit rate (%)", "L2 Hit Rate", "%", None),
        ("Achieved occupancy (%)", "Achieved Occupancy", "%", None),
        ("Registers per thread", "Registers Per Thread", "register/thread", None),
    ]
    rows: list[dict[str, float | str]] = []
    for label, metric_name, unit, section in specifications:
        row: dict[str, float | str] = {"Metric": label}
        for phase in phases:
            selected = ncu[
                (ncu["phase"] == phase)
                & (ncu["Metric Name"] == metric_name)
                & (ncu["Metric Unit"] == unit)
            ]
            if section is not None:
                selected = selected[selected["Section Name"] == section]
            if len(selected) != 1:
                raise ValueError(
                    f"Expected one NCU value for {phase}/{metric_name}/{unit}, "
                    f"got {len(selected)}"
                )
            row[phase] = float(selected["Metric Value"].iloc[0])
        rows.append(row)
    return pd.DataFrame(rows)



def cross_gpu_table(cross_paper: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        cross_paper,
        [
            "gpu",
            "architecture",
            "compute_capability",
            "pytorch_total_ms",
            "warp_total_ms",
            "warp_total_between_trial_std_ms",
            "warp_speedup_vs_pytorch",
            "warp_reduction_vs_pytorch_pct",
            "warp_w",
            "warp_bwdk_block",
        ],
        "Three-GPU paper table",
    )
    if list(cross_paper["gpu"]) != GPU_ORDER:
        cross_paper = cross_paper.set_index("gpu").loc[GPU_ORDER].reset_index()
    result = pd.DataFrame(
        {
            "GPU": cross_paper["gpu"].map(GPU_SHORT),
            "Architecture": cross_paper["architecture"],
            "CC": cross_paper["compute_capability"],
            "PyTorch total (ms)": cross_paper["pytorch_total_ms"],
            "Warp-tiled total (ms)": cross_paper["warp_total_ms"],
            "Warp-tiled std (ms)": cross_paper[
                "warp_total_between_trial_std_ms"
            ],
            "Speedup": cross_paper["warp_speedup_vs_pytorch"],
            "Reduction (%)": cross_paper["warp_reduction_vs_pytorch_pct"],
            "W": cross_paper["warp_w"].astype(int),
            "dK block": cross_paper["warp_bwdk_block"].astype(int),
        }
    )
    return result



def compile_table(cross: pd.DataFrame, compile_data: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        compile_data,
        ["trials", "total_mean_ms", "total_between_trial_std_ms"],
        "torch.compile aggregate",
    )
    a6000 = cross[cross["gpu"] == GPU_ORDER[1]].set_index("method")
    eager_total = float(a6000.loc["PyTorch", "total_mean_ms"])
    eager_std = float(a6000.loc["PyTorch", "total_between_trial_std_ms"])
    warp_total = float(a6000.loc["Warp", "total_mean_ms"])
    warp_std = float(a6000.loc["Warp", "total_between_trial_std_ms"])
    compile_total = float(compile_data["total_mean_ms"].iloc[0])
    compile_std = float(compile_data["total_between_trial_std_ms"].iloc[0])
    trials = int(compile_data["trials"].iloc[0])
    return pd.DataFrame(
        [
            {
                "Backend": "PyTorch eager",
                "Trials": 3,
                "Total latency (ms)": eager_total,
                "Between-trial std (ms)": eager_std,
                "Warp-tiled speedup": eager_total / warp_total,
            },
            {
                "Backend": "torch.compile",
                "Trials": trials,
                "Total latency (ms)": compile_total,
                "Between-trial std (ms)": compile_std,
                "Warp-tiled speedup": compile_total / warp_total,
            },
            {
                "Backend": "Warp-tiled",
                "Trials": 3,
                "Total latency (ms)": warp_total,
                "Between-trial std (ms)": warp_std,
                "Warp-tiled speedup": 1.0,
            },
        ]
    )



def external_baseline_table(external_paper: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        external_paper,
        [
            "gpu",
            "pytorch_total_ms",
            "triton_total_ms",
            "tvm_tirx_total_ms",
            "pathtile_total_ms",
            "pathtile_speedup_vs_triton",
            "pathtile_speedup_vs_tvm_tirx",
            "pathtile_speedup_vs_best_compiler",
            "tvm_evidence_status",
        ],
        "External compiler paper table",
    )
    gpu_order = ["A100", "RTX A6000", "V100"]
    indexed = external_paper.set_index("gpu").loc[gpu_order]
    evidence = indexed["tvm_evidence_status"].map(
        {
            "PASS": "Pass",
            "QUALIFIED_WITH_CAVEAT": "Caveat",
        }
    )
    if evidence.isna().any():
        raise ValueError("Unknown TVM evidence status")
    return pd.DataFrame(
        {
            "GPU": gpu_order,
            "PyTorch (ms)": indexed["pytorch_total_ms"].to_numpy(),
            "Triton (ms)": indexed["triton_total_ms"].to_numpy(),
            "TVM-TIRx (ms)": indexed["tvm_tirx_total_ms"].to_numpy(),
            "PathTile (ms)": indexed["pathtile_total_ms"].to_numpy(),
            "PT / best compiler": indexed[
                "pathtile_speedup_vs_best_compiler"
            ].to_numpy(),
            "TVM evidence": evidence.to_numpy(),
        }
    )



def main() -> None:
    args = parse_args()
    root = args.project_root.expanduser().resolve()
    output_root = args.output_dir.expanduser()
    if not output_root.is_absolute():
        output_root = root / output_root
    figures_dir = output_root / "figures"
    tables_dir = output_root / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    configure_matplotlib()
    cross = read_csv(
        root,
        "results/cross_gpu_final_summary/cross_gpu_3gpu_all_metrics.csv",
    )
    require_columns(
        cross,
        [
            "gpu",
            "method",
            "fwd_mean_ms",
            "bwd_input_mean_ms",
            "bwd_weight_mean_ms",
            "total_mean_ms",
            "total_between_trial_std_ms",
            "fwd_speedup_vs_pytorch",
            "bwd_input_speedup_vs_pytorch",
            "bwd_weight_speedup_vs_pytorch",
            "total_speedup_vs_pytorch",
        ],
        "Three-GPU all metrics",
    )
    if set(cross["gpu"]) != set(GPU_ORDER) or set(cross["method"]) != set(METHOD_ORDER):
        raise ValueError("Three-GPU aggregate does not contain the expected GPUs/methods")
    cross_paper_source = read_csv(
        root,
        "results/cross_gpu_final_summary/cross_gpu_3gpu_paper_table.csv",
    )
    holdout = read_csv(
        root,
        "results/pathtile_selector/holdout_evaluation/three_gpu_holdout_per_shape.csv",
    )
    holdout_summary = read_csv(
        root,
        "results/pathtile_selector/holdout_evaluation/three_gpu_holdout_summary.csv",
    )
    ncu = read_csv(
        root,
        "results/nsight/a100_main/a100_main_ncu_key_metrics.csv",
    )
    compile_data = read_csv(
        root,
        "results/a6000_final_baselines/torch_compile/final_aggregate.csv",
    )
    external = read_csv(
        root,
        "results/external_baselines/final_summary/external_baselines_all_metrics.csv",
    )
    external_paper = read_csv(
        root,
        "results/external_baselines/final_summary/external_baselines_paper_table.csv",
    )
    generated: list[Path] = []
    generated.extend(cross_gpu_figure_detailed(cross, figures_dir))
    generated.extend(v100_ablation_figure_detailed(cross, figures_dir))
    generated.extend(
        external_baseline_figure_detailed(external, external_paper, figures_dir)
    )
    shape_data = prepare_shape_data(root)
    shape_paths, shape_summary = shape_generalization_figure_detailed(
        shape_data, figures_dir
    )
    generated.extend(shape_paths)
    generated.extend(write_table(shape_summary, tables_dir / "table_shape_generalization"))
    generated.extend(holdout_figure_detailed(holdout, figures_dir))
    generated.extend(write_table(holdout_summary, tables_dir / "table_holdout_summary"))
    e2e_paths, e2e_table, e2e_summary = end_to_end_figure_and_table_detailed(
        root, figures_dir
    )
    generated.extend(e2e_paths)
    generated.extend(write_table(e2e_table, tables_dir / "table_end_to_end_a6000"))
    generated.extend(write_table(e2e_summary, tables_dir / "table_end_to_end_speedups"))
    generated.extend(write_table(cross_gpu_table(cross_paper_source), tables_dir / "table_cross_gpu"))
    generated.extend(write_table(ncu_table(ncu), tables_dir / "table_ncu_a100"))
    generated.extend(
        write_table(compile_table(cross, compile_data), tables_dir / "table_torch_compile_a6000")
    )
    generated.extend(
        write_table(
            external_baseline_table(external_paper),
            tables_dir / "table_external_baselines",
        )
    )
    manifest = pd.DataFrame(
        {
            "artifact": [str(path.relative_to(output_root)) for path in generated],
            "kind": [path.suffix.lstrip(".") for path in generated],
        }
    )
    manifest_path = output_root / "artifact_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    print("Final paper artifacts: PASS")
    print(f"Output directory: {output_root}")
    print(f"Generated files: {len(generated) + 1}")
    print(manifest.to_string(index=False))
    print(f"{manifest_path.relative_to(output_root)}  csv")

if __name__ == "__main__":
    main()
