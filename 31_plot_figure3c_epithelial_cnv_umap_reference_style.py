"""Build Figure 3C: epithelial HCNV/LCNV and continuous CNV-score UMAPs.

The script reuses the exact Figure 3A embedding and Figure 3B expression-CNV
scores. HCNV/LCNV is a median split, matching the relative high/low grouping
shown by the reference panel; it is not a DNA-level aneuploid/euploid call.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 0.8,
    }
)

ROOT = Path(__file__).resolve().parent
FIG3A_CELLS = ROOT / "results/paper9/figure3_epithelial/Fig3A_epithelial_cells_and_umap.csv.gz"
FIG3B_SCORES = ROOT / "results/paper9/figure3_epithelial/Fig3B_epithelial_cnv_scores.csv.gz"
RESULT_DIR = ROOT / "results/paper9/figure3_epithelial"
FIGURE_DIR = ROOT / "figures/paper9/figure3_epithelial"
OUTPUT_STEM = FIGURE_DIR / "Fig3C_epithelial_cnv_umap_reference_style"
SEED = 20260831

GROUP_COLORS = {"HCNV": "#1F77B4", "LCNV": "#FF7F0E"}
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "reference_cnv_score",
    [
        (0.00, "#D5D5D5"),
        (0.68, "#BDBDC1"),
        (0.84, "#9D83A5"),
        (1.00, "#6D1D68"),
    ],
)


def load_cells() -> tuple[pd.DataFrame, float]:
    cells = pd.read_csv(FIG3A_CELLS)
    scores = pd.read_csv(FIG3B_SCORES)
    required_a = {
        "cell_id",
        "umap_1",
        "umap_2",
        "epithelial_cluster",
        "disease_status",
        "patient_id",
        "sample_id",
    }
    required_b = {"cell_id", "cnv_score"}
    if not required_a.issubset(cells.columns):
        raise RuntimeError(f"Figure 3A table lacks {sorted(required_a - set(cells.columns))}")
    if not required_b.issubset(scores.columns):
        raise RuntimeError(f"Figure 3B table lacks {sorted(required_b - set(scores.columns))}")
    merged = cells.merge(
        scores[["cell_id", "cnv_score"]], on="cell_id", how="left", validate="one_to_one"
    )
    if merged["cnv_score"].isna().any():
        raise RuntimeError("Some Figure 3A cells do not have Figure 3B CNV scores")
    threshold = float(merged["cnv_score"].median())
    merged["cnv_group"] = np.where(merged["cnv_score"] > threshold, "HCNV", "LCNV")
    return merged, threshold


def robust_limits(embedding: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_low, x_high = np.quantile(embedding[:, 0], [0.001, 0.999])
    y_low, y_high = np.quantile(embedding[:, 1], [0.001, 0.999])
    x_pad = max((x_high - x_low) * 0.04, 0.1)
    y_pad = max((y_high - y_low) * 0.04, 0.1)
    return (x_low - x_pad, x_high + x_pad), (y_low - y_pad, y_high + y_pad)


def axis_glyph(ax: plt.Axes) -> None:
    color = "#222222"
    origin = (0.040, 0.058)
    ax.plot(
        [origin[0], origin[0] + 0.108],
        [origin[1], origin[1]],
        transform=ax.transAxes,
        color=color,
        lw=0.85,
        clip_on=False,
    )
    ax.plot(
        [origin[0], origin[0]],
        [origin[1], origin[1] + 0.130],
        transform=ax.transAxes,
        color=color,
        lw=0.85,
        clip_on=False,
    )
    ax.plot(
        [origin[0], origin[0] - 0.026],
        [origin[1], origin[1] - 0.020],
        transform=ax.transAxes,
        color=color,
        lw=0.85,
        clip_on=False,
    )
    ax.text(
        origin[0] + 0.057,
        origin[1] - 0.045,
        "X_umap1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.5,
    )
    ax.text(
        origin[0] - 0.030,
        origin[1] + 0.070,
        "X_umap2",
        transform=ax.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=7.5,
    )


def format_umap(
    ax: plt.Axes,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    axis_glyph(ax)


def group_legend() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=6.0,
            markerfacecolor=color,
            markeredgecolor="none",
            label=group,
        )
        for group, color in GROUP_COLORS.items()
    ]


def plot_figure(cells: pd.DataFrame) -> tuple[float, float]:
    embedding = cells[["umap_1", "umap_2"]].to_numpy(dtype=float)
    groups = cells["cnv_group"].astype(str).to_numpy()
    scores = cells["cnv_score"].to_numpy(dtype=float)
    xlim, ylim = robust_limits(embedding)
    rng = np.random.default_rng(SEED)
    random_order = rng.permutation(len(cells))

    score_vmin = 0.0
    score_vmax = float(np.quantile(scores, 0.95))
    score_order = np.argsort(scores, kind="mergesort")

    fig = plt.figure(figsize=(7.15, 3.25), facecolor="white")
    ax_group = fig.add_axes([0.055, 0.070, 0.350, 0.835])
    ax_score = fig.add_axes([0.535, 0.070, 0.350, 0.835])

    for group, color in GROUP_COLORS.items():
        selected = random_order[groups[random_order] == group]
        ax_group.scatter(
            embedding[selected, 0],
            embedding[selected, 1],
            s=3.0,
            c=color,
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )
    format_umap(ax_group, xlim, ylim)
    ax_group.legend(
        handles=group_legend(),
        loc="center left",
        bbox_to_anchor=(1.005, 0.61),
        frameon=False,
        fontsize=8.2,
        handlelength=0.7,
        handletextpad=0.35,
        labelspacing=0.48,
        borderaxespad=0,
    )

    ax_score.scatter(
        embedding[score_order, 0],
        embedding[score_order, 1],
        s=3.0,
        c=scores[score_order],
        cmap=SCORE_CMAP,
        vmin=score_vmin,
        vmax=score_vmax,
        alpha=0.82,
        linewidths=0,
        rasterized=True,
    )
    format_umap(ax_score, xlim, ylim)
    ax_score.set_title("cnv_score", fontsize=8.0, pad=3)

    color_axis = fig.add_axes([0.887, 0.205, 0.016, 0.330])
    colorbar = fig.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=score_vmin, vmax=score_vmax), cmap=SCORE_CMAP
        ),
        cax=color_axis,
    )
    candidate_ticks = np.asarray([0.02, 0.04, 0.06, 0.08])
    ticks = candidate_ticks[candidate_ticks < score_vmax + 1e-9]
    colorbar.set_ticks(ticks)
    colorbar.ax.tick_params(labelsize=6.8, width=0.7, length=2.8)
    colorbar.outline.set_linewidth(0.75)

    fig.text(0.012, 0.965, "C", fontsize=23, fontweight="normal", va="top", ha="left")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    return score_vmin, score_vmax


def export_results(
    cells: pd.DataFrame,
    threshold: float,
    score_vmin: float,
    score_vmax: float,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_columns = [
        "cell_id",
        "patient_id",
        "sample_id",
        "disease_status",
        "epithelial_cluster",
        "umap_1",
        "umap_2",
        "cnv_score",
        "cnv_group",
    ]
    cells[output_columns].to_csv(
        RESULT_DIR / "Fig3C_epithelial_cnv_groups_and_umap.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    cluster_summary = (
        cells.groupby(["epithelial_cluster", "cnv_group"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    cluster_summary["fraction_within_cluster"] = cluster_summary["n_cells"] / cluster_summary.groupby(
        "epithelial_cluster", observed=False
    )["n_cells"].transform("sum")
    cluster_summary.to_csv(
        RESULT_DIR / "Fig3C_epithelial_cnv_group_by_cluster.csv",
        index=False,
        encoding="utf-8-sig",
    )
    stage_summary = (
        cells.groupby(["disease_status", "cnv_group"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    stage_summary["fraction_within_stage"] = stage_summary["n_cells"] / stage_summary.groupby(
        "disease_status", observed=False
    )["n_cells"].transform("sum")
    stage_summary.to_csv(
        RESULT_DIR / "Fig3C_epithelial_cnv_group_by_stage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "figure3a_cell_table": str(FIG3A_CELLS),
        "figure3b_score_table": str(FIG3B_SCORES),
        "n_cells": int(len(cells)),
        "cnv_group_definition": "HCNV if cnv_score > epithelial median; otherwise LCNV",
        "cnv_score_median_threshold": threshold,
        "hcnv_n": int(cells["cnv_group"].eq("HCNV").sum()),
        "lcnv_n": int(cells["cnv_group"].eq("LCNV").sum()),
        "continuous_color_limits": [score_vmin, score_vmax],
        "continuous_color_upper_rule": "epithelial cnv_score 95th percentile",
        "random_seed": SEED,
        "visual_reference": "s41746-025-02297-4 Figure 3C",
        "interpretation_boundary": (
            "HCNV and LCNV are relative expression-derived CNV-score groups. "
            "They must not be described as DNA-validated aneuploid and euploid cells."
        ),
    }
    (RESULT_DIR / "Fig3C_epithelial_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    print("Loading Figure 3A embedding and Figure 3B CNV scores...", flush=True)
    cells, threshold = load_cells()
    print(
        f"  cells={len(cells):,}; median threshold={threshold:.6f}; "
        f"HCNV={cells['cnv_group'].eq('HCNV').sum():,}; "
        f"LCNV={cells['cnv_group'].eq('LCNV').sum():,}",
        flush=True,
    )
    score_vmin, score_vmax = plot_figure(cells)
    export_results(cells, threshold, score_vmin, score_vmax)
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}", flush=True)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
