"""Plot Figure 2D: cell2location abundance maps for the Figure 2A section."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd


INPUT = ROOT / "data" / "processed" / "GSE208253" / "GSE208253_cleaned_deconvolved.h5ad"
OUTPUT_DIR = ROOT / "figures" / "paper9" / "figure2"
SELECTION = OUTPUT_DIR / "Fig2A_representative_sample_selection.csv"
OUTPUT = OUTPUT_DIR / "Fig2D_cell2location_abundance_early_OSCC"
SCALE_AUDIT = OUTPUT_DIR / "Fig2D_abundance_scale_limits.csv"
PANEL_LETTER = "D"
PANEL_TITLE = "Abundance-early OSCC"

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

# Keep the reference panel order while substituting the current paper9 classes.
PANEL_ORDER = [
    "NK/T cells",
    "Epithelial cells",
    "Endothelial cells",
    "Fibroblasts",
    "Macrophages",
    "B/Plasma cells",
    "Myocytes",
    "Dendritic cells",
    "Mast cells",
]

LABELS = {
    "B/Plasma cells": "B/Plasma",
    "Dendritic cells": "Dendritic",
    "Endothelial cells": "Endothelial",
    "Epithelial cells": "Epithelial",
    "Fibroblasts": "Fibro",
    "Macrophages": "Macrophage",
    "Mast cells": "Mast",
    "Myocytes": "Myocytes",
    "NK/T cells": "T_NK",
}

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


def abundance_cmap(hex_color: str) -> colors.LinearSegmentedColormap:
    rgb = np.asarray(colors.to_rgb(hex_color))
    baseline = 0.76 * np.ones(3) + 0.24 * rgb
    midpoint = 0.38 * np.ones(3) + 0.62 * rgb
    peak = np.clip(rgb * 0.72, 0, 1)
    return colors.LinearSegmentedColormap.from_list(
        "cell2location_abundance",
        [tuple(baseline), tuple(midpoint), tuple(peak)],
        N=256,
    )


def darker(hex_color: str, factor: float = 0.64) -> tuple[float, float, float]:
    return tuple(np.clip(np.asarray(colors.to_rgb(hex_color)) * factor, 0, 1))


def main() -> None:
    if not SELECTION.exists():
        raise FileNotFoundError(f"Run Figure 2A first; missing {SELECTION}")
    selection = pd.read_csv(SELECTION)
    chosen = selection.loc[selection["selected"].astype(bool), "sample_id"].astype(str).tolist()
    if len(chosen) != 1:
        raise ValueError("Figure 2A selection audit must contain exactly one selected sample")
    sample_id = chosen[0]

    adata = ad.read_h5ad(INPUT)
    selected = adata.obs["sample_id"].astype(str).eq(sample_id).to_numpy()
    coordinates = np.asarray(adata.obsm["spatial"])[selected]
    abundance = pd.DataFrame(
        np.asarray(adata.obsm["cell2location_abundance"])[selected],
        columns=CELL_TYPES,
    )

    x_limits = padded_limits(coordinates[:, 0])
    y_limits = padded_limits(coordinates[:, 1])[::-1]
    fig = plt.figure(figsize=(2.48, 2.34), facecolor="white")
    grid = fig.add_gridspec(
        3,
        3,
        left=0.07,
        right=0.985,
        bottom=0.025,
        top=0.84,
        wspace=0.08,
        hspace=0.15,
    )
    axes = [fig.add_subplot(grid[row, column]) for row in range(3) for column in range(3)]

    scale_rows: list[dict[str, object]] = []
    scalefactors = adata.uns["spatial"][sample_id]["scalefactors"]
    pixels_per_micron = float(scalefactors["spot_diameter_fullres"]) / 55.0
    scale_length = 500.0 * pixels_per_micron
    scale_fraction = scale_length / (x_limits[1] - x_limits[0])

    for ax, cell_type in zip(axes, PANEL_ORDER):
        values = abundance[cell_type].to_numpy(dtype=float)
        vmin = float(np.quantile(values, 0.01))
        vmax = float(np.quantile(values, 0.99))
        if vmax <= vmin:
            vmax = float(values.max())
        order = np.argsort(values, kind="stable")
        ax.scatter(
            coordinates[order, 0],
            coordinates[order, 1],
            c=values[order],
            cmap=abundance_cmap(CELL_COLORS[cell_type]),
            vmin=vmin,
            vmax=vmax,
            s=0.90,
            linewidths=0,
            edgecolors="none",
            rasterized=True,
        )
        ax.set_xlim(x_limits)
        ax.set_ylim(y_limits)
        ax.set_aspect("equal", adjustable="box")
        ax.axis("off")
        ax.set_title(
            LABELS[cell_type],
            fontsize=5.0,
            fontweight="normal",
            color="#202020",
            pad=0.35,
        )

        scale_start = 0.87 - scale_fraction
        scale_y = 0.10
        ax.add_line(
            Line2D(
                [scale_start, 0.87],
                [scale_y, scale_y],
                transform=ax.transAxes,
                color="white",
                linewidth=1.50,
                solid_capstyle="butt",
                clip_on=False,
                zorder=20,
            )
        )
        ax.add_line(
            Line2D(
                [scale_start, 0.87],
                [scale_y, scale_y],
                transform=ax.transAxes,
                color="#242424",
                linewidth=0.65,
                solid_capstyle="butt",
                clip_on=False,
                zorder=21,
            )
        )
        scale_text = ax.text(
            0.87,
            scale_y + 0.035,
            r"500 $\mu$m",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=2.55,
            color="#242424",
            zorder=21,
        )
        scale_text.set_path_effects(
            [path_effects.withStroke(linewidth=0.65, foreground="white")]
        )
        scale_rows.append(
            {
                "sample_id": sample_id,
                "cell_type": cell_type,
                "display_vmin_q01": vmin,
                "display_vmax_q99": vmax,
                "raw_min": float(values.min()),
                "raw_max": float(values.max()),
            }
        )

    strip = Rectangle(
        (0.14, 0.895),
        0.845,
        0.078,
        transform=fig.transFigure,
        facecolor="#D9D9DD",
        edgecolor="none",
        clip_on=False,
        zorder=10,
    )
    fig.add_artist(strip)
    fig.text(
        0.562,
        0.934,
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scale_rows).to_csv(SCALE_AUDIT, index=False)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=600, facecolor="white")
    plt.close(fig)
    print(f"Sample {sample_id}; 9 cell2location abundance maps")
    print(f"Saved {OUTPUT.with_suffix('.png')}")
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")
    print(f"Saved {SCALE_AUDIT}")


if __name__ == "__main__":
    main()
