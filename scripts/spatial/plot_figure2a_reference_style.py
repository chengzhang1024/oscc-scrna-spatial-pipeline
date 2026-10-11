"""Plot Figure 2A: representative early-stage OSCC cell-type map."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


INPUT = ROOT / "data" / "processed" / "GSE208253" / "GSE208253_cleaned_deconvolved.h5ad"
OUTPUT_DIR = ROOT / "figures" / "paper9" / "figure2"
OUTPUT = OUTPUT_DIR / "Fig2A_cell2location_early_OSCC"
AUDIT = OUTPUT_DIR / "Fig2A_representative_sample_selection.csv"
PANEL_LETTER = "A"
PANEL_TITLE = "cell2location-early OSCC"

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

# Palette established for the current Figure 1 from the supplied references.
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

LEGEND_LABELS = {
    "B/Plasma cells": "B/Plasma",
    "Dendritic cells": "DC",
    "Endothelial cells": "Endo",
    "Epithelial cells": "Epithelial",
    "Fibroblasts": "Fibro",
    "Macrophages": "Macro",
    "Mast cells": "Mast",
    "Myocytes": "Myocyte",
    "NK/T cells": "NK/T",
}


def choose_representative(
    obs: pd.DataFrame, proportions: pd.DataFrame
) -> tuple[str, pd.DataFrame]:
    early = obs["stage"].astype(str).isin(["Stage I", "Stage II"])
    sample_means = proportions.loc[early].groupby(obs.loc[early, "sample_id"].astype(str)).mean()

    # Aitchison distance is appropriate for comparing cell-type compositions.
    clr = np.log(sample_means.clip(lower=1e-6))
    clr = clr.sub(clr.mean(axis=1), axis=0)
    centroid = clr.mean(axis=0)
    distances = np.sqrt(clr.sub(centroid, axis=1).pow(2).sum(axis=1))

    metadata = (
        obs.loc[early, ["sample_id", "patient_id", "stage"]]
        .assign(sample_id=lambda frame: frame["sample_id"].astype(str))
        .drop_duplicates("sample_id")
        .set_index("sample_id")
    )
    audit = metadata.join(distances.rename("aitchison_distance_to_early_centroid"))
    audit = audit.join(sample_means.add_prefix("mean_"))
    audit = audit.sort_values("aitchison_distance_to_early_centroid")
    audit.insert(0, "selected", False)
    audit.iloc[0, audit.columns.get_loc("selected")] = True
    return str(audit.index[0]), audit.reset_index()


def padded_limits(values: np.ndarray, fraction: float = 0.025) -> tuple[float, float]:
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    padding = max((upper - lower) * fraction, 1.0)
    return lower - padding, upper + padding


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(INPUT)
    proportions = pd.DataFrame(
        np.asarray(adata.obsm["cell2location_proportions"]),
        index=adata.obs_names,
        columns=CELL_TYPES,
    )
    sample_id, audit = choose_representative(adata.obs, proportions)
    audit.to_csv(AUDIT, index=False)

    selected = adata.obs["sample_id"].astype(str).eq(sample_id)
    sample_obs = adata.obs.loc[selected]
    sample_proportions = proportions.loc[selected, CELL_TYPES]
    coordinates = np.asarray(adata.obsm["spatial"])[selected.to_numpy()]
    dominant = sample_proportions.idxmax(axis=1).to_numpy()

    fig = plt.figure(figsize=(2.35, 2.42), facecolor="white")
    ax = fig.add_axes([0.105, 0.205, 0.86, 0.64])

    order = np.argsort(sample_proportions.max(axis=1).to_numpy(), kind="stable")
    ax.scatter(
        coordinates[order, 0],
        coordinates[order, 1],
        c=[CELL_COLORS[cell_type] for cell_type in dominant[order]],
        s=5.2,
        alpha=0.93,
        linewidths=0,
        edgecolors="none",
        rasterized=True,
    )
    ax.set_xlim(padded_limits(coordinates[:, 0]))
    ax.set_ylim(padded_limits(coordinates[:, 1])[::-1])
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    ax.add_patch(
        Rectangle(
            (0, 1.015),
            1,
            0.13,
            transform=ax.transAxes,
            facecolor="#D9D9DD",
            edgecolor="none",
            clip_on=False,
            zorder=10,
        )
    )
    ax.text(
        0.5,
        1.08,
        PANEL_TITLE,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=7.4,
        fontweight="bold",
        fontstyle="italic",
        color="#202020",
        zorder=11,
    )

    scalefactors = adata.uns["spatial"][sample_id]["scalefactors"]
    pixels_per_micron = float(scalefactors["spot_diameter_fullres"]) / 55.0
    scale_length = 500.0 * pixels_per_micron
    x_min, x_max = ax.get_xlim()
    scale_fraction = scale_length / (x_max - x_min) * 0.86
    scale_x = 0.13
    scale_y = 0.075
    fig.add_artist(
        Line2D(
            [scale_x, scale_x + scale_fraction],
            [scale_y, scale_y],
            transform=fig.transFigure,
            color="#111111",
            linewidth=2.0,
            solid_capstyle="butt",
            clip_on=False,
        )
    )
    fig.text(
        scale_x + scale_fraction / 2,
        scale_y + 0.027,
        r"500 $\mu$m",
        ha="center",
        va="bottom",
        fontsize=5.2,
        color="#111111",
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=3.4,
            markerfacecolor=CELL_COLORS[cell_type],
            markeredgecolor="none",
            label=LEGEND_LABELS[cell_type],
        )
        for cell_type in CELL_TYPES
    ]
    legend_box = Rectangle(
        (0.405, 0.025),
        0.565,
        0.145,
        transform=fig.transFigure,
        facecolor="#D9D9DD",
        edgecolor="none",
        clip_on=False,
        zorder=1,
    )
    fig.add_artist(legend_box)
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.688, 0.098),
        ncol=3,
        frameon=False,
        fontsize=3.65,
        handletextpad=0.20,
        columnspacing=0.55,
        labelspacing=0.20,
        borderaxespad=0,
        markerscale=0.82,
    )
    fig.text(
        0.012, 0.978, PANEL_LETTER, ha="left", va="top", fontsize=11.5, color="#202020"
    )

    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=600, facecolor="white")
    plt.close(fig)

    selected_meta = sample_obs.iloc[0]
    print(
        f"Selected {sample_id} | {selected_meta['patient_id']} | {selected_meta['stage']} "
        f"| {selected.sum():,} spots"
    )
    print(f"Saved {OUTPUT.with_suffix('.png')}")
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")
    print(f"Saved {AUDIT}")


if __name__ == "__main__":
    main()
