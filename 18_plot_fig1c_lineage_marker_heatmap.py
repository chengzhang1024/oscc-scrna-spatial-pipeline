"""Build Figure 1C as a balanced single-cell lineage marker heatmap."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
import numpy as np
from scipy import sparse


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
OUTPUT = ROOT / "figures/paper9/Fig1C_lineage_marker_heatmap"
CELLS_PER_TYPE = 100
RANDOM_SEED = 181919

# The order follows the reference figure's tumor/immune/stromal visual flow.
CELL_MARKERS = {
    "NK/T cells": ["CD3D", "TRAC", "NKG7"],
    "Epithelial cells": ["KRT19", "KRT5", "KRT14"],
    "Endothelial cells": ["VWF", "CLDN5", "PECAM1"],
    "Fibroblasts": ["LUM", "DCN", "COL1A1"],
    "Macrophages": ["C1QA", "LYZ", "CD68"],
    "Dendritic cells": ["HLA-DRA", "CD1C", "FCER1A"],
    "B/Plasma cells": ["CD79A", "MS4A1", "MZB1"],
    "Mast cells": ["TPSAB1", "TPSB2", "CPA3"],
    "Myocytes": ["ACTA1", "ACTN2", "MYL2"],
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


def extract_selected_rows(
    h5: h5py.File,
    cell_indices: np.ndarray,
    requested_genes: list[str],
) -> np.ndarray:
    genes = decode(h5["var"]["_index"][:])
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    missing = [gene for gene in requested_genes if gene not in gene_to_index]
    if missing:
        raise KeyError(f"Genes missing from integrated object: {missing}")
    selected_genes = np.asarray([gene_to_index[gene] for gene in requested_genes], dtype=np.int32)

    n_cells, n_genes = (int(value) for value in h5["X"].attrs["shape"])
    indptr = h5["X"]["indptr"][:]
    expression = np.zeros((len(cell_indices), len(selected_genes)), dtype=np.float32)
    for output_row, cell_index in enumerate(cell_indices):
        if cell_index < 0 or cell_index >= n_cells:
            raise IndexError(cell_index)
        start, end = int(indptr[cell_index]), int(indptr[cell_index + 1])
        row = sparse.csr_matrix(
            (
                h5["X"]["data"][start:end],
                h5["X"]["indices"][start:end],
                np.asarray([0, end - start], dtype=np.int64),
            ),
            shape=(1, n_genes),
        )
        expression[output_row] = row[:, selected_genes].toarray()[0]
    return expression


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    cell_types = list(CELL_MARKERS)
    marker_genes = [gene for markers in CELL_MARKERS.values() for gene in markers]

    with h5py.File(H5AD, "r") as h5:
        labels = read_categorical(h5["obs"], "cell_type_paper9")
        sampled_by_type: dict[str, np.ndarray] = {}
        for cell_type in cell_types:
            available = np.flatnonzero(labels == cell_type)
            n_sample = min(CELLS_PER_TYPE, len(available))
            sampled_by_type[cell_type] = np.sort(rng.choice(available, n_sample, replace=False))
        sampled_indices = np.concatenate([sampled_by_type[cell_type] for cell_type in cell_types])
        expression = extract_selected_rows(h5, sampled_indices, marker_genes)

    # Heatmap rows are genes and columns are cells. Scaling each gene makes
    # lineage specificity visible without comparing absolute gene abundance.
    matrix = expression.T.astype(float)
    gene_mean = matrix.mean(axis=1, keepdims=True)
    gene_std = matrix.std(axis=1, keepdims=True)
    scaled = np.clip((matrix - gene_mean) / (gene_std + 1e-8), -2.5, 2.5)

    # Use the narrow, portrait-like footprint of the reference panel so this
    # marker overview can sit beside other Figure 1 panels without dominating
    # the horizontal space.
    fig = plt.figure(figsize=(5.4, 7.2), facecolor="white")
    grid = fig.add_gridspec(
        3,
        2,
        width_ratios=[1, 0.72],
        height_ratios=[1, 0.022, 0.16],
        left=0.07,
        right=0.97,
        bottom=0.065,
        top=0.91,
        wspace=0.025,
        hspace=0.025,
    )
    heat_ax = fig.add_subplot(grid[0, 0])
    label_ax = fig.add_subplot(grid[0, 1], sharey=heat_ax)
    strip_ax = fig.add_subplot(grid[1, 0], sharex=heat_ax)
    legend_ax = fig.add_subplot(grid[2, :])

    heatmap_cmap = colors.LinearSegmentedColormap.from_list(
        "article_heatmap", ["#3F78A8", "#F5F5F5", "#CF535E"]
    )
    image = heat_ax.imshow(
        scaled,
        aspect="auto",
        interpolation="nearest",
        cmap=heatmap_cmap,
        vmin=-2.5,
        vmax=2.5,
        rasterized=True,
    )
    heat_ax.set_xticks([])
    heat_ax.set_yticks([])
    for spine in heat_ax.spines.values():
        spine.set_visible(False)

    rows_per_type = 3
    for block_index, cell_type in enumerate(cell_types):
        start = block_index * rows_per_type
        center = start + (rows_per_type - 1) / 2
        if block_index:
            heat_ax.axhline(start - 0.5, color="#222222", linestyle=(0, (3, 2)), linewidth=0.7)
        label_ax.text(
            0.02,
            center,
            ", ".join(CELL_MARKERS[cell_type]),
            ha="left",
            va="center",
            fontsize=7.2,
            color="#303030",
        )
    label_ax.set_xlim(0, 1)
    label_ax.set_ylim(len(marker_genes) - 0.5, -0.5)
    label_ax.axis("off")

    annotation_values = np.concatenate(
        [np.full(len(sampled_by_type[cell_type]), index) for index, cell_type in enumerate(cell_types)]
    )[None, :]
    strip_cmap = colors.ListedColormap([CELL_COLORS[cell_type] for cell_type in cell_types])
    strip_ax.imshow(annotation_values, aspect="auto", interpolation="nearest", cmap=strip_cmap)
    strip_ax.set_xticks([])
    strip_ax.set_yticks([])
    for spine in strip_ax.spines.values():
        spine.set_visible(False)

    legend_ax.axis("off")
    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="none",
            markersize=6,
            markerfacecolor=CELL_COLORS[cell_type],
            markeredgecolor="none",
            label=cell_type.replace(" cells", ""),
        )
        for cell_type in cell_types
    ]
    legend_ax.legend(
        handles=handles,
        ncol=3,
        loc="upper center",
        frameon=False,
        fontsize=7.2,
        handletextpad=0.35,
        columnspacing=0.75,
        borderaxespad=0,
    )

    color_ax = fig.add_axes([0.62, 0.936, 0.26, 0.014])
    colorbar = fig.colorbar(image, cax=color_ax, orientation="horizontal")
    colorbar.set_ticks([-2.5, 0, 2.5])
    colorbar.set_ticklabels(["Low", "", "High"])
    colorbar.ax.tick_params(labelsize=7, length=0)
    colorbar.ax.set_title("Scaled expression", fontsize=7.5, pad=3)
    colorbar.outline.set_visible(False)

    fig.text(0.015, 0.955, "C", fontsize=22, ha="left", va="top")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.with_suffix('.png')}")
    print(f"Saved: {OUTPUT.with_suffix('.pdf')}")
    print(f"Cells per type: {CELLS_PER_TYPE}; total cells: {len(sampled_indices)}")


if __name__ == "__main__":
    main()
