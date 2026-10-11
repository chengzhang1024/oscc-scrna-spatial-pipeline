"""Plot Figure 2B: spatial marker maps for the Figure 2A section."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import sparse


INPUT = ROOT / "data" / "processed" / "GSE208253" / "GSE208253_cleaned_deconvolved.h5ad"
OUTPUT_DIR = ROOT / "figures" / "paper9" / "figure2"
SELECTION = OUTPUT_DIR / "Fig2A_representative_sample_selection.csv"
OUTPUT = OUTPUT_DIR / "Fig2B_markers_early_OSCC"
PANEL_LETTER = "B"
PANEL_TITLE = "Markers-early OSCC"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

MARKERS = [
    ("CD3D", "NK/T cells"),
    ("EPCAM", "Epithelial cells"),
    ("PECAM1", "Endothelial cells"),
    ("COL1A1", "Fibroblasts"),
    ("CD68", "Macrophages"),
    ("MS4A1", "B/Plasma cells"),
    ("RGS5", "Myocytes"),
    ("CD8A", "NK/T cells"),
    ("CD163", "Macrophages"),
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


def padded_limits(values: np.ndarray, fraction: float = 0.02) -> tuple[float, float]:
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    padding = max((upper - lower) * fraction, 1.0)
    return lower - padding, upper + padding


def expression_cmap(hex_color: str) -> colors.LinearSegmentedColormap:
    rgb = np.asarray(colors.to_rgb(hex_color))
    baseline = 0.80 * np.ones(3) + 0.20 * rgb
    midpoint = 0.55 * np.ones(3) + 0.45 * rgb
    return colors.LinearSegmentedColormap.from_list(
        "marker_expression",
        [tuple(baseline), tuple(midpoint), hex_color],
        N=256,
    )


def darker(hex_color: str, factor: float = 0.72) -> tuple[float, float, float]:
    rgb = np.asarray(colors.to_rgb(hex_color))
    return tuple(np.clip(rgb * factor, 0, 1))


def main() -> None:
    if not SELECTION.exists():
        raise FileNotFoundError(f"Run Figure 2A first; missing {SELECTION}")
    selection = pd.read_csv(SELECTION)
    chosen = selection.loc[selection["selected"].astype(bool), "sample_id"].astype(str).tolist()
    if len(chosen) != 1:
        raise ValueError("Figure 2A selection audit must contain exactly one selected sample")
    sample_id = chosen[0]

    adata = ad.read_h5ad(INPUT)
    genes = [gene for gene, _ in MARKERS]
    missing = [gene for gene in genes if gene not in adata.var_names]
    if missing:
        raise KeyError(f"Marker genes absent from spatial object: {', '.join(missing)}")

    selected = adata.obs["sample_id"].astype(str).eq(sample_id).to_numpy()
    coordinates = np.asarray(adata.obsm["spatial"])[selected]
    matrix = adata[selected, genes].X
    if sparse.issparse(matrix):
        matrix = matrix.toarray()
    matrix = np.asarray(matrix, dtype=float)

    x_limits = padded_limits(coordinates[:, 0])
    y_limits = padded_limits(coordinates[:, 1])[::-1]
    fig = plt.figure(figsize=(2.48, 2.62), facecolor="white")
    grid = fig.add_gridspec(
        3,
        3,
        left=0.055,
        right=0.905,
        bottom=0.035,
        top=0.855,
        wspace=0.22,
        hspace=0.16,
    )
    axes = [fig.add_subplot(grid[row, column]) for row in range(3) for column in range(3)]

    for index, (ax, (gene, cell_type)) in enumerate(zip(axes, MARKERS)):
        values = matrix[:, index]
        vmax = max(float(np.quantile(values, 0.99)), float(values.max()) * 0.25, 1e-3)
        order = np.argsort(values, kind="stable")
        points = ax.scatter(
            coordinates[order, 0],
            coordinates[order, 1],
            c=values[order],
            cmap=expression_cmap(CELL_COLORS[cell_type]),
            vmin=0,
            vmax=vmax,
            s=0.78,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(
            gene,
            fontsize=6.5,
            fontweight="bold",
            fontstyle="italic",
            color=darker(CELL_COLORS[cell_type]),
            pad=0.6,
        )

        position = ax.get_position()
        color_ax = fig.add_axes(
            [position.x1 + 0.004, position.y0 + position.height * 0.20, 0.008, position.height * 0.43]
        )
        colorbar = fig.colorbar(points, cax=color_ax, orientation="vertical")
        colorbar.set_ticks([0, vmax / 2, vmax])
        colorbar.ax.set_yticklabels(["0", f"{vmax / 2:.1f}", f"{vmax:.1f}"])
        colorbar.ax.tick_params(labelsize=1.9, length=0.75, width=0.3, pad=0.5)
        colorbar.outline.set_visible(False)

    strip = Rectangle(
        (0.14, 0.905),
        0.83,
        0.075,
        transform=fig.transFigure,
        facecolor="#D9D9DD",
        edgecolor="none",
        clip_on=False,
        zorder=10,
    )
    fig.add_artist(strip)
    fig.text(
        0.555,
        0.942,
        PANEL_TITLE,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold",
        fontstyle="italic",
        color="#202020",
        zorder=11,
    )
    fig.text(
        0.012, 0.982, PANEL_LETTER, ha="left", va="top", fontsize=11.5, color="#202020"
    )

    scalefactors = adata.uns["spatial"][sample_id]["scalefactors"]
    pixels_per_micron = float(scalefactors["spot_diameter_fullres"]) / 55.0
    scale_length = 500.0 * pixels_per_micron
    scale_fraction = scale_length / (x_limits[1] - x_limits[0]) * axes[2].get_position().width
    scale_x = 0.945 - scale_fraction
    scale_y = 0.867
    fig.add_artist(
        Line2D(
            [scale_x, scale_x + scale_fraction],
            [scale_y, scale_y],
            transform=fig.transFigure,
            color="#111111",
            linewidth=1.25,
            solid_capstyle="butt",
            clip_on=False,
        )
    )
    fig.text(
        scale_x + scale_fraction / 2,
        scale_y + 0.008,
        r"500 $\mu$m",
        ha="center",
        va="bottom",
        fontsize=2.8,
        color="#111111",
    )

    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Sample {sample_id}; markers: {', '.join(genes)}")
    print(f"Saved {OUTPUT.with_suffix('.png')}")
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
