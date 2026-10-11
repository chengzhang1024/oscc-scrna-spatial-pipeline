"""Build Figure 1D-F: cell-type similarity and patient-level TME dynamics."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage


matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
    }
)

ROOT = Path(__file__).resolve().parent
H5AD = ROOT / "results/integrated_3ds_annotated_paper9.h5ad"
ROE_CSV = ROOT / "results/paper9/roe_literature_v3_GSE181919_Normal_OLK_OSCC.csv"
COMPOSITION_CSV = ROOT / "results/paper9/composition_literature_v3_GSE181919_per_sample.csv"
OUTPUT = ROOT / "figures/paper9/Fig1DEF_TME_dynamics"
D_OUTPUT = ROOT / "figures/paper9/Fig1D_stage_enriched_similarity_tree"
F_OUTPUT = ROOT / "figures/paper9/Fig1F_stage_fraction_lollipops"

CELL_TYPES = [
    "B/Plasma cells",
    "Dendritic cells",
    "Endothelial cells",
    "Epithelial cells",
    "Fibroblasts",
    "Macrophages",
    "Mast cells",
    "Myocytes",
    "NK/T cells",
]
SUPPORTED_TYPES = [
    "B/Plasma cells",
    "Dendritic cells",
    "Endothelial cells",
    "Epithelial cells",
    "Fibroblasts",
    "Macrophages",
    "NK/T cells",
]
CELL_COLORS = {
    "B/Plasma cells": "#65BBA0",
    "Dendritic cells": "#F9DF95",
    "Endothelial cells": "#8AC9BE",
    "Epithelial cells": "#CF535E",
    "Fibroblasts": "#F6B273",
    "Macrophages": "#5CA48E",
    "Mast cells": "#D3304F",
    "Myocytes": "#E76C43",
    "NK/T cells": "#F7B52B",
}
GROUPS = ["Normal", "OLK", "OSCC"]
GROUP_COLORS = {"Normal": "#5F8AA8", "OLK": "#E87A55", "OSCC": "#5CA48E"}


def decode(values: np.ndarray) -> np.ndarray:
    return np.asarray(
        [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in values],
        dtype=object,
    )


def read_categorical(group: h5py.Group, name: str) -> np.ndarray:
    item = group[name]
    categories = decode(item["categories"][:])
    codes = item["codes"][:]
    output = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    output[valid] = categories[codes[valid]]
    return output


def celltype_pca_centroids() -> np.ndarray:
    with h5py.File(H5AD, "r") as h5:
        labels = read_categorical(h5["obs"], "cell_type_paper9")
        pca = h5["obsm"]["X_pca"][:, :30]
    return np.vstack([pca[labels == cell_type].mean(axis=0) for cell_type in CELL_TYPES])


def sample_balanced_roe_all_types(composition: pd.DataFrame) -> pd.DataFrame:
    group_mean = composition.groupby("disease_status", observed=True)[CELL_TYPES].mean()
    expected = composition[CELL_TYPES].mean(axis=0)
    return group_mean.div(expected.replace(0, np.nan), axis=1).reindex(index=GROUPS, columns=CELL_TYPES)


def polar_segment(ax: plt.Axes, x: np.ndarray, y: np.ndarray, max_distance: float) -> None:
    n_leaves = len(CELL_TYPES)

    def theta(value: np.ndarray | float) -> np.ndarray:
        return 2 * np.pi * np.asarray(value) / (10 * n_leaves)

    def radius(value: np.ndarray | float) -> np.ndarray:
        return 1.0 - 0.78 * np.asarray(value) / max(max_distance, 1e-8)

    parent_y = y[1]
    ax.plot([theta(x[0]), theta(x[1])], [radius(y[0]), radius(parent_y)], color="#202020", lw=1.5)
    arc_x = np.linspace(x[1], x[2], 80)
    ax.plot(theta(arc_x), np.full_like(arc_x, radius(parent_y)), color="#202020", lw=1.5)
    ax.plot([theta(x[2]), theta(x[3])], [radius(parent_y), radius(y[3])], color="#202020", lw=1.5)


def plot_similarity_tree(ax: plt.Axes, stage_roe: pd.DataFrame) -> None:
    centroids = celltype_pca_centroids()
    tree = linkage(centroids, method="ward", metric="euclidean")
    result = dendrogram(tree, labels=CELL_TYPES, no_plot=True)
    cluster_ids = fcluster(tree, t=4, criterion="maxclust")
    cluster_by_label = dict(zip(CELL_TYPES, cluster_ids))
    max_distance = max(max(values) for values in result["dcoord"])
    leaf_order = result["ivl"]
    n_leaves = len(leaf_order)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Shade contiguous clades from the hierarchical clustering, matching the
    # reference figure's colored sector backgrounds.
    clade_colors = ["#E7F2EC", "#E7F0F5", "#F7ECE6", "#FBF4D9"]
    runs: list[tuple[int, int, int]] = []
    run_start = 0
    current_cluster = cluster_by_label[leaf_order[0]]
    for index in range(1, n_leaves + 1):
        next_cluster = cluster_by_label[leaf_order[index]] if index < n_leaves else None
        if next_cluster != current_cluster:
            runs.append((run_start, index - 1, current_cluster))
            run_start = index
            current_cluster = next_cluster
    unique_clusters = list(dict.fromkeys(cluster_id for _, _, cluster_id in runs))
    cluster_color = {
        cluster_id: clade_colors[index % len(clade_colors)]
        for index, cluster_id in enumerate(unique_clusters)
    }
    for start, end, cluster_id in runs:
        width = (end - start + 1) * 2 * np.pi / n_leaves * 0.96
        center = (start + end + 1) * np.pi / n_leaves
        ax.bar(
            center,
            0.80,
            width=width,
            bottom=0.20,
            color=cluster_color[cluster_id],
            edgecolor="none",
            alpha=0.78,
            zorder=0,
        )

    for x, y in zip(result["icoord"], result["dcoord"]):
        polar_segment(ax, np.asarray(x), np.asarray(y), max_distance)

    stage_scale = 0.050
    maximum_stack = max(
        sum(min(float(stage_roe.loc[group, label]), 4.0) * stage_scale for group in GROUPS)
        for label in leaf_order
    )
    for index, label in enumerate(leaf_order):
        theta = 2 * np.pi * (index + 0.5) / n_leaves
        display_angle = ((90 - np.degrees(theta) + 180) % 360) - 180
        rotation = ((display_angle - 90 + 180) % 360) - 180
        if rotation > 90:
            rotation -= 180
        elif rotation < -90:
            rotation += 180
        sector_width = 2 * np.pi / n_leaves * 0.58
        stack_bottom = 1.015
        for group in GROUPS:
            value = float(stage_roe.loc[group, label])
            bar_height = min(value, 4.0) * stage_scale
            ax.bar(
                theta,
                bar_height,
                width=sector_width,
                bottom=stack_bottom,
                color=GROUP_COLORS[group],
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
            stack_bottom += bar_height
        ax.text(
            theta,
            1.015 + maximum_stack + 0.085,
            label.replace(" cells", ""),
            rotation=rotation,
            rotation_mode="anchor",
            ha="center",
            va="center",
            fontsize=8.5,
            color="#222222",
        )
    ax.set_ylim(0, 1.015 + maximum_stack + 0.25)
    ax.set_axis_off()
    stage_handles = [Patch(facecolor=GROUP_COLORS[group], edgecolor="none", label=group) for group in GROUPS]
    ax.legend(
        handles=stage_handles,
        title="Group",
        loc="center left",
        bbox_to_anchor=(1.00, 0.50),
        frameon=False,
        fontsize=7.5,
        title_fontsize=8,
        handlelength=1.0,
        handletextpad=0.45,
        borderaxespad=0,
    )


def save_standalone_tree(stage_roe: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(7.4, 7.0), facecolor="white")
    axis = figure.add_subplot(111, projection="polar")
    plot_similarity_tree(axis, stage_roe)
    figure.text(0.025, 0.965, "D", fontsize=24, ha="left", va="top")
    figure.savefig(D_OUTPUT.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    figure.savefig(D_OUTPUT.with_suffix(".pdf"), dpi=400, bbox_inches="tight", pad_inches=0.10, facecolor="white")
    plt.close(figure)


def plot_roe_heatmap(ax: plt.Axes, roe: pd.DataFrame) -> None:
    roe = roe.reindex(index=GROUPS, columns=CELL_TYPES)
    log_roe = np.log2(roe.astype(float))
    limit = max(2.0, float(np.nanmax(np.abs(log_roe.to_numpy()))))
    cmap = colors.LinearSegmentedColormap.from_list(
        "roe_diverging", ["#79BFB1", "#FAF7F2", "#CF535E"]
    )
    image = ax.imshow(log_roe, cmap=cmap, vmin=-limit, vmax=limit, aspect="auto")
    ax.set_xticks(
        range(len(CELL_TYPES)),
        [name.replace(" cells", "") for name in CELL_TYPES],
        rotation=90,
        ha="center",
        va="top",
        fontsize=6.7,
    )
    ax.set_yticks(range(len(GROUPS)), GROUPS, fontsize=8.5)
    ax.tick_params(length=0)
    for y, group in enumerate(GROUPS):
        for x, cell_type in enumerate(CELL_TYPES):
            value = float(roe.loc[group, cell_type])
            text_color = "white" if abs(float(log_roe.loc[group, cell_type])) > 1.15 else "#333333"
            ax.text(x, y, f"{value:.2f}", ha="center", va="center", fontsize=7.2, color=text_color)
    for spine in ax.spines.values():
        spine.set_edgecolor("#777777")
        spine.set_linewidth(0.8)
    ax.set_title("Tissue prevalence of cell types", fontsize=9.2, weight="normal", pad=7)
    colorbar = ax.figure.colorbar(image, ax=ax, orientation="vertical", fraction=0.040, pad=0.025, aspect=15)
    colorbar.set_label("log2(Ro/e)", fontsize=6.7, labelpad=2)
    colorbar.ax.tick_params(labelsize=7, length=2)
    colorbar.outline.set_visible(False)


def plot_fraction_panel(ax: plt.Axes, data: pd.DataFrame, cell_type: str) -> None:
    medians = []
    for group in GROUPS:
        values = data.loc[data["disease_status"] == group, cell_type].astype(float).to_numpy()
        medians.append(float(np.median(values)))

    # Reference-style lollipops: a neutral stem encodes magnitude while the
    # stage-colored endpoint makes cross-stage comparison immediate.
    maximum = max(medians) if medians else 1.0
    upper_limit = max(0.012, maximum * 1.34)
    for x, group in enumerate(GROUPS):
        value = medians[x]
        ax.vlines(x, 0, value, color="#777777", linewidth=2.1, zorder=1)
        ax.scatter(
            x,
            value,
            s=48,
            color=GROUP_COLORS[group],
            edgecolor="#333333",
            linewidth=0.75,
            zorder=3,
        )
        value_label = f"{value:.3f}" if value < 0.1 else f"{value:.2f}"
        ax.text(
            x,
            value + upper_limit * 0.035,
            value_label,
            ha="center",
            va="bottom",
            fontsize=6.2,
            color="#333333",
            clip_on=False,
        )

    ax.set_xlim(-0.35, 2.35)
    ax.set_ylim(0, upper_limit)
    ax.set_xticks(range(3), GROUPS, rotation=90, fontsize=6.5)
    ax.tick_params(axis="x", length=0, pad=1)
    ax.tick_params(axis="y", labelsize=6.5, length=2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_linewidth(0.7)
    ax.spines[["left", "bottom"]].set_color("#444444")
    ax.add_patch(
        Rectangle(
            (0, 1.015),
            1,
            0.12,
            transform=ax.transAxes,
            facecolor=CELL_COLORS[cell_type],
            edgecolor="#555555",
            linewidth=0.5,
            clip_on=False,
        )
    )
    ax.text(
        0.5,
        1.075,
        cell_type.replace(" cells", ""),
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7,
        color="#222222",
    )


def save_standalone_lollipops(composition: pd.DataFrame) -> None:
    figure = plt.figure(figsize=(6.0, 5.3), facecolor="white")
    grid = figure.add_gridspec(
        3,
        3,
        left=0.10,
        right=0.985,
        bottom=0.105,
        top=0.90,
        wspace=0.45,
        hspace=0.62,
    )
    axes = [figure.add_subplot(grid[row, column]) for row in range(3) for column in range(3)]
    for axis, cell_type in zip(axes, CELL_TYPES):
        plot_fraction_panel(axis, composition, cell_type)
    figure.text(0.018, 0.965, "F", fontsize=22, ha="left", va="top")
    axes[3].set_ylabel("Median cell fraction", fontsize=7.2, labelpad=3)
    figure.savefig(F_OUTPUT.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    figure.savefig(F_OUTPUT.with_suffix(".pdf"), dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(figure)


def main() -> None:
    composition = pd.read_csv(COMPOSITION_CSV)
    composition = composition[composition["disease_status"].isin(GROUPS)].copy()
    stage_roe_all = sample_balanced_roe_all_types(composition)

    # Match the reference Figure 1D-F footprint: D/E share the wider left
    # column and the compact lollipop matrix spans the full right column.
    fig = plt.figure(figsize=(10.0, 7.2), facecolor="white")
    outer = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.20, 0.92],
        height_ratios=[1.20, 0.80],
        left=0.065,
        right=0.985,
        bottom=0.10,
        top=0.925,
        wspace=0.24,
        hspace=0.30,
    )
    tree_ax = fig.add_subplot(outer[0, 0], projection="polar")
    roe_ax = fig.add_subplot(outer[1, 0])
    fraction_grid = outer[:, 1].subgridspec(3, 3, wspace=0.43, hspace=0.58)
    fraction_axes = [
        fig.add_subplot(fraction_grid[row, column]) for row in range(3) for column in range(3)
    ]

    plot_similarity_tree(tree_ax, stage_roe_all)
    plot_roe_heatmap(roe_ax, stage_roe_all)
    for ax, cell_type in zip(fraction_axes, CELL_TYPES):
        plot_fraction_panel(ax, composition, cell_type)
    fraction_axes[3].set_ylabel("Median cell fraction", fontsize=6.7, labelpad=2)

    fig.text(0.015, 0.965, "D", fontsize=22, ha="left", va="top")
    fig.text(0.015, 0.420, "E", fontsize=22, ha="left", va="top")
    fig.text(0.575, 0.965, "F", fontsize=22, ha="left", va="top")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=400, facecolor="white")
    plt.close(fig)
    save_standalone_tree(stage_roe_all)
    save_standalone_lollipops(composition)
    print(f"Saved: {OUTPUT.with_suffix('.png')}")
    print(f"Saved: {OUTPUT.with_suffix('.pdf')}")
    print(f"Saved: {D_OUTPUT.with_suffix('.png')}")
    print(f"Saved: {D_OUTPUT.with_suffix('.pdf')}")
    print(f"Saved: {F_OUTPUT.with_suffix('.png')}")
    print(f"Saved: {F_OUTPUT.with_suffix('.pdf')}")
    print("D: PCA-profile similarity; E: sample-balanced Ro/e; F: stage median fractions")


RANDOM_SEED = 181919


if __name__ == "__main__":
    main()
