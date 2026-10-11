"""Plot integrated Normal, OLK, and OSCC cells in a compact reference style."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np


ROOT = Path(__file__).resolve().parent
H5AD = ROOT / "results/integrated_3ds_annotated_paper9.h5ad"
OUTPUT = ROOT / "figures/paper9/Normal_OLK_OSCC_celltypes_reference_style"

STAGES = ["Normal", "OLK", "OSCC"]
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

# Sampled from the supplied article figures, then expanded within the same
# coral/orange/green palette to give all nine atlas classes distinct colors.
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


def padded_limits(values: np.ndarray, fraction: float = 0.025) -> tuple[float, float]:
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    padding = (upper - lower) * fraction
    return lower - padding, upper + padding


def add_title_strip(ax: plt.Axes, title: str) -> None:
    ax.add_patch(
        Rectangle(
            (0, 1.015),
            1,
            0.105,
            transform=ax.transAxes,
            facecolor="#EEEFF4",
            edgecolor="none",
            clip_on=False,
            zorder=10,
        )
    )
    ax.text(
        0.5,
        1.067,
        title,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="#252525",
        zorder=11,
    )


def main() -> None:
    with h5py.File(H5AD, "r") as h5:
        disease = read_categorical(h5["obs"], "disease_status")
        cell_type = read_categorical(h5["obs"], "cell_type_paper9")
        umap = h5["obsm"]["X_umap"][:]

    plotted = np.isin(disease, STAGES) & np.isin(cell_type, CELL_TYPES)
    x_limits = padded_limits(umap[plotted, 0])
    y_limits = padded_limits(umap[plotted, 1])

    fig = plt.figure(figsize=(12.6, 3.25), facecolor="white")
    grid = fig.add_gridspec(
        1,
        4,
        width_ratios=[1, 1, 1, 0.62],
        left=0.025,
        right=0.985,
        bottom=0.055,
        top=0.91,
        wspace=0.045,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    legend_ax = fig.add_subplot(grid[0, 3])

    rng = np.random.default_rng(181919)
    for ax, stage in zip(axes, STAGES):
        indices = np.flatnonzero(plotted & (disease == stage))
        indices = rng.permutation(indices)
        point_colors = [CELL_COLORS[label] for label in cell_type[indices]]
        ax.scatter(
            umap[indices, 0],
            umap[indices, 1],
            s=0.7,
            c=point_colors,
            alpha=0.78,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        add_title_strip(ax, stage)

    legend_ax.axis("off")
    handles = [Patch(facecolor=CELL_COLORS[label], edgecolor="none", label=label) for label in CELL_TYPES]
    legend_ax.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.02, 0.52),
        frameon=False,
        fontsize=8.5,
        handlelength=0.9,
        handleheight=0.9,
        handletextpad=0.55,
        labelspacing=0.62,
        borderaxespad=0,
    )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=400, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    counts = {
        stage: int(np.count_nonzero(plotted & (disease == stage)))
        for stage in STAGES
    }
    print(f"Saved: {OUTPUT.with_suffix('.png')}")
    print(f"Saved: {OUTPUT.with_suffix('.pdf')}")
    print(f"Plotted cells: {counts}")


if __name__ == "__main__":
    main()
