"""Build Figure 3F: epithelial pathway activity heatmap.

Activity is calculated from Figure 3E values by z-scoring each gene's cluster
mean across C0-C8 and averaging the z-scores within each prespecified program.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
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
FIG3E_VALUES = ROOT / "results/paper9/figure3_epithelial/Fig3E_epithelial_pathway_dotplot_values.csv"
RESULT_DIR = ROOT / "results/paper9/figure3_epithelial"
FIGURE_DIR = ROOT / "figures/paper9/figure3_epithelial"
OUTPUT_STEM = FIGURE_DIR / "Fig3F_epithelial_pathway_activity_reference_style"
CLUSTERS = [f"C{i}" for i in range(9)]
PROGRAMS = [
    "Keratinization",
    "EMT",
    "IFN / inflammation",
    "Hypoxia / glycolysis",
    "DNA repair",
    "Cell cycle",
]
DISPLAY_LABELS = [
    "Keratinization",
    "EMT",
    "IFN /\ninflammation",
    "Hypoxia /\nglycolysis",
    "DNA\nrepair",
    "Cell\ncycle",
]
VMIN = -2.0
VMAX = 2.0

ACTIVITY_CMAP = LinearSegmentedColormap.from_list(
    "pathway_activity",
    [
        (0.00, "#08306B"),
        (0.25, "#6BAED6"),
        (0.50, "#F7F7F7"),
        (0.75, "#FCAE91"),
        (1.00, "#B2182B"),
    ],
)


def calculate_activity() -> tuple[pd.DataFrame, pd.DataFrame]:
    values = pd.read_csv(FIG3E_VALUES)
    required = {
        "epithelial_cluster",
        "program",
        "gene",
        "mean_log_expression",
    }
    if not required.issubset(values.columns):
        raise RuntimeError(f"Figure 3E values lack {sorted(required - set(values.columns))}")
    if set(values["epithelial_cluster"].astype(str)) != set(CLUSTERS):
        raise RuntimeError("Figure 3E does not contain exactly C0-C8")
    if set(values["program"].astype(str)) != set(PROGRAMS):
        raise RuntimeError("Figure 3E programs do not match the Figure 3F plan")

    values = values.copy()
    values["gene_mean_across_clusters"] = values.groupby("gene", observed=False)[
        "mean_log_expression"
    ].transform("mean")
    values["gene_sd_across_clusters"] = values.groupby("gene", observed=False)[
        "mean_log_expression"
    ].transform(lambda series: series.std(ddof=0))
    values["gene_cluster_zscore"] = (
        values["mean_log_expression"] - values["gene_mean_across_clusters"]
    ) / (values["gene_sd_across_clusters"] + 1e-6)

    activity = (
        values.groupby(["epithelial_cluster", "program"], observed=False)[
            "gene_cluster_zscore"
        ]
        .mean()
        .unstack("program")
        .reindex(index=CLUSTERS, columns=PROGRAMS)
    )
    if activity.isna().any().any():
        raise RuntimeError("Pathway activity matrix contains missing values")
    return activity, values


def plot_figure(activity: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    matrix = activity.to_numpy(dtype=float)
    fig = plt.figure(figsize=(6.65, 4.95), facecolor="white")
    ax = fig.add_axes([0.155, 0.205, 0.665, 0.650])

    ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=ACTIVITY_CMAP,
        vmin=VMIN,
        vmax=VMAX,
        origin="upper",
        rasterized=True,
    )
    ax.set_xticks(np.arange(len(PROGRAMS)))
    ax.set_xticklabels(
        DISPLAY_LABELS,
        fontsize=7.2,
        fontweight="bold",
        fontstyle="italic",
        linespacing=0.92,
    )
    ax.set_yticks(np.arange(len(CLUSTERS)))
    ax.set_yticklabels(CLUSTERS, fontsize=8.2)
    ax.set_ylabel("Cell type", fontsize=9.0, labelpad=8)
    ax.tick_params(axis="x", width=0.8, length=3.2, pad=7)
    ax.tick_params(axis="y", width=0.8, length=3.2, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            color = "white" if abs(value) >= 1.35 else "#262626"
            ax.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7.3,
                color=color,
            )

    ax.set_title("Pathway activity per cell type", fontsize=8.8, pad=8)

    color_axis = fig.add_axes([0.855, 0.205, 0.038, 0.650])
    colorbar = fig.colorbar(
        ScalarMappable(
            norm=Normalize(vmin=VMIN, vmax=VMAX), cmap=ACTIVITY_CMAP
        ),
        cax=color_axis,
    )
    colorbar.set_ticks([-2, -1, 0, 1, 2])
    colorbar.ax.tick_params(labelsize=7.2, width=0.75, length=3.0, pad=3)
    colorbar.outline.set_visible(False)

    fig.text(0.020, 0.965, "F", fontsize=25, fontweight="normal", va="top", ha="left")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def export_results(activity: pd.DataFrame, gene_values: pd.DataFrame) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    activity.rename_axis("epithelial_cluster").reset_index().to_csv(
        RESULT_DIR / "Fig3F_epithelial_pathway_activity.csv",
        index=False,
        encoding="utf-8-sig",
    )
    gene_values[
        [
            "epithelial_cluster",
            "program",
            "gene",
            "mean_log_expression",
            "gene_mean_across_clusters",
            "gene_sd_across_clusters",
            "gene_cluster_zscore",
        ]
    ].to_csv(
        RESULT_DIR / "Fig3F_epithelial_pathway_gene_zscores.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "figure3e_values": str(FIG3E_VALUES),
        "clusters": CLUSTERS,
        "programs": PROGRAMS,
        "matrix_shape": [len(CLUSTERS), len(PROGRAMS)],
        "method": (
            "For each gene, z-score cluster mean log-normalized expression across "
            "C0-C8; pathway activity is the arithmetic mean of member-gene z-scores."
        ),
        "genes_per_program": 7,
        "display_color_limits": [VMIN, VMAX],
        "visual_reference": "s41746-025-02297-4 Figure 3F",
        "user_decision": "C5 and C8 retained without exclusion",
        "interpretation_boundary": (
            "Activities are cluster-level descriptive expression scores, not "
            "patient-balanced differential pathway tests. C8 is immune-like and "
            "C5 is mixed, so their activity profiles require the existing QC caveat."
        ),
    }
    (RESULT_DIR / "Fig3F_epithelial_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    print("Calculating Figure 3F pathway activity from Figure 3E...", flush=True)
    activity, gene_values = calculate_activity()
    print(activity.round(2).to_string(), flush=True)
    export_results(activity, gene_values)
    plot_figure(activity)
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}", flush=True)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
