"""Plot Figure 2C from patient-aggregated Early-stage MISTy results."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


INPUT = ROOT / "results" / "spatial_misty" / "figure2c_early" / "early_misty_group_importance.csv"
OUTPUT = ROOT / "figures" / "paper9" / "figure2" / "Fig2C_MISTy_early_OSCC"
PANEL_LETTER = "C"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

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

LABELS = {
    "B/Plasma cells": "B/Plasma",
    "Dendritic cells": "D_C",
    "Endothelial cells": "Endothelial",
    "Epithelial cells": "Epithelial",
    "Fibroblasts": "Fibro",
    "Macrophages": "Macrophage",
    "Mast cells": "Mast",
    "Myocytes": "Myocyte",
    "NK/T cells": "T_NK",
}

VIEWS = [
    ("intra", "Misty-intra"),
    ("juxta", "Misty-juxta_5"),
    ("para", "Misty-para_15"),
]

STABILITY_MIN = 0.75
IMPORTANCE_MIN = 1.00


def main() -> None:
    data = pd.read_csv(INPUT)
    cmap = colors.LinearSegmentedColormap.from_list(
        "paper9_misty_green", ["#EEF6F2", "#65BBA0", "#2F7D68"], N=256
    )
    cmap.set_bad("white")

    fig = plt.figure(figsize=(7.45, 2.68), facecolor="white")
    grid = fig.add_gridspec(
        1,
        3,
        left=0.065,
        right=0.975,
        bottom=0.235,
        top=0.82,
        wspace=0.42,
    )
    axes = [fig.add_subplot(grid[0, index]) for index in range(3)]
    predictor_order = CELL_TYPES
    target_order = list(reversed(CELL_TYPES))

    for ax, (view, title) in zip(axes, VIEWS):
        subset = data.loc[data["view"].eq(view)].copy()
        subset["display"] = subset["median_importance_z"].where(
            subset["stability_fraction"].ge(STABILITY_MIN)
            & subset["median_importance_z"].ge(IMPORTANCE_MIN)
        )
        matrix = (
            subset.pivot(index="target", columns="predictor", values="display")
            .reindex(index=target_order, columns=predictor_order)
            .to_numpy(dtype=float)
        )
        masked = np.ma.masked_invalid(matrix)
        image = ax.imshow(masked, cmap=cmap, vmin=IMPORTANCE_MIN, vmax=2.5, aspect="equal")
        ax.set_xticks(np.arange(len(predictor_order)))
        ax.set_yticks(np.arange(len(target_order)))
        ax.set_xticklabels(
            [LABELS[cell_type] for cell_type in predictor_order],
            rotation=48,
            ha="right",
            rotation_mode="anchor",
            fontsize=5.0,
        )
        ax.set_yticklabels([LABELS[cell_type] for cell_type in target_order], fontsize=5.0)
        ax.tick_params(axis="both", length=0, pad=1.2, colors="#202020")
        ax.set_xticks(np.arange(-0.5, len(predictor_order), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(target_order), 1), minor=True)
        ax.grid(
            which="minor",
            color="#1F1F1F",
            linestyle=(0, (3, 2)),
            linewidth=0.55,
        )
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_visible(False)

        position = ax.get_position()
        strip = Rectangle(
            (position.x0 - 0.018, 0.875),
            position.width + 0.036,
            0.095,
            transform=fig.transFigure,
            facecolor="#D9D9DD",
            edgecolor="none",
            clip_on=False,
            zorder=10,
        )
        fig.add_artist(strip)
        fig.text(
            position.x0 + position.width / 2,
            0.922,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold",
            fontstyle="italic",
            color="#202020",
            zorder=11,
        )

        color_ax = fig.add_axes(
            [position.x1 + 0.010, position.y0 + position.height * 0.28, 0.010, position.height * 0.42]
        )
        colorbar = fig.colorbar(image, cax=color_ax, orientation="vertical")
        colorbar.set_ticks([1.0, 1.5, 2.0, 2.5])
        colorbar.ax.tick_params(labelsize=4.5, length=1.2, width=0.4, pad=1.0)
        colorbar.ax.set_title("Importance", fontsize=4.7, pad=2.5)
        colorbar.outline.set_visible(False)

    fig.text(
        0.008, 0.978, PANEL_LETTER, ha="left", va="top", fontsize=13.5, color="#202020"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Saved {OUTPUT.with_suffix('.png')}")
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
