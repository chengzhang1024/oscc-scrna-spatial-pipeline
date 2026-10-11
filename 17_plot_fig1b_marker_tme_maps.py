"""Build Figure 1B: marker audit plus four major TME compartment maps."""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors, transforms
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
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
MARKER_STATS = ROOT / "results/paper9_marker_stats.csv"
OUTPUT = ROOT / "figures/paper9/Fig1B_marker_TME_feature_maps"

CELL_MARKERS = {
    "B/Plasma cells": ["CD79A", "MS4A1", "MZB1"],
    "Dendritic cells": ["CD1C", "FCER1A", "HLA-DRA"],
    "Endothelial cells": ["PECAM1", "VWF", "CLDN5"],
    "Epithelial cells": ["EPCAM", "KRT5", "KRT14", "KRT19"],
    "Fibroblasts": ["COL1A1", "DCN", "LUM"],
    "Macrophages": ["C1QA", "CD68", "LYZ"],
    "Mast cells": ["TPSAB1", "TPSB2", "CPA3"],
    "Myocytes": ["ACTA1", "ACTN2", "MYL2"],
    "NK/T cells": ["CD3D", "TRAC", "NKG7"],
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

FEATURES = {
    "KRT19": "Tumor epithelial",
    "LUM": "Fibroblast stroma",
    "VWF": "Vascular stroma",
    "PTPRC": "Immune compartment",
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


def extract_expression(
    h5: h5py.File,
    requested: list[str],
    chunk_size: int = 5000,
) -> dict[str, np.ndarray]:
    genes = decode(h5["var"]["_index"][:])
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    missing = [gene for gene in requested if gene not in gene_to_index]
    if missing:
        raise KeyError(f"Genes missing from integrated object: {missing}")

    selected = np.asarray([gene_to_index[gene] for gene in requested], dtype=np.int32)
    n_cells, n_genes = (int(value) for value in h5["X"].attrs["shape"])
    output = np.zeros((n_cells, len(selected)), dtype=np.float32)
    indptr = h5["X"]["indptr"][:]
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        data_start, data_end = int(indptr[start]), int(indptr[end])
        chunk = sparse.csr_matrix(
            (
                h5["X"]["data"][data_start:data_end],
                h5["X"]["indices"][data_start:data_end],
                indptr[start : end + 1] - data_start,
            ),
            shape=(end - start, n_genes),
        )
        output[start:end] = chunk[:, selected].toarray()
    return {gene: output[:, index] for index, gene in enumerate(requested)}


def make_dotplot(ax: plt.Axes) -> ScalarMappable:
    cell_types = list(CELL_MARKERS)
    genes = [gene for markers in CELL_MARKERS.values() for gene in markers]
    stats = pd.read_csv(MARKER_STATS)
    stats = stats[(stats["dataset"] == "ALL") & stats["gene"].isin(genes)]
    means = stats.pivot(index="cell_type", columns="gene", values="mean_expression").reindex(
        index=cell_types, columns=genes
    )
    expressed = stats.pivot(index="cell_type", columns="gene", values="pct_expressed").reindex(
        index=cell_types, columns=genes
    )

    scaled = means.copy()
    for gene in genes:
        values = means[gene].to_numpy(float)
        scaled[gene] = (values - values.mean()) / (values.std() + 1e-8)

    marker_cmap = colors.LinearSegmentedColormap.from_list(
        "marker_purple", ["#F5F4F7", "#B9A8D2", "#5B3F8C"]
    )
    norm = colors.Normalize(vmin=-1.0, vmax=2.5)
    for y, cell_type in enumerate(cell_types):
        for x, gene in enumerate(genes):
            ax.scatter(
                x,
                y,
                s=7 + 1.7 * float(expressed.loc[cell_type, gene]),
                c=[marker_cmap(norm(float(scaled.loc[cell_type, gene])))],
                edgecolors="#6F6680",
                linewidths=0.25,
                zorder=3,
            )

    start = 0
    blended = transforms.blended_transform_factory(ax.transData, ax.transAxes)
    for cell_type, markers in CELL_MARKERS.items():
        width = len(markers)
        ax.add_patch(
            Rectangle(
                (start - 0.5, -0.5),
                width,
                len(cell_types),
                fill=False,
                edgecolor="#D6D3DA",
                linewidth=0.7,
                zorder=1,
            )
        )
        ax.add_patch(
            Rectangle(
                (start - 0.5, 1.018),
                width,
                0.027,
                transform=blended,
                facecolor=CELL_COLORS[cell_type],
                edgecolor="none",
                clip_on=False,
            )
        )
        start += width

    ax.set_xlim(-0.8, len(genes) - 0.5)
    ax.set_ylim(len(cell_types) - 0.5, -0.5)
    ax.set_xticks(range(len(genes)), genes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(cell_types)), [name.replace(" cells", "") for name in cell_types], fontsize=8.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for spine in ax.spines.values():
        spine.set_visible(False)

    size_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=np.sqrt(7 + 1.7 * pct),
            markerfacecolor="white",
            markeredgecolor="#6F6680",
            markeredgewidth=0.5,
        )
        for pct in (25, 50, 75)
    ]
    ax.legend(
        size_handles,
        ["25", "50", "75"],
        title="Percent expressed",
        ncol=3,
        frameon=False,
        fontsize=7,
        title_fontsize=7.5,
        handletextpad=0.2,
        columnspacing=0.7,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.055),
        borderaxespad=0,
    )
    return ScalarMappable(norm=norm, cmap=marker_cmap)


def padded_limits(values: np.ndarray, fraction: float = 0.025) -> tuple[float, float]:
    lower, upper = float(values.min()), float(values.max())
    padding = (upper - lower) * fraction
    return lower - padding, upper + padding


def plot_feature(
    ax: plt.Axes,
    umap: np.ndarray,
    values: np.ndarray,
    included: np.ndarray,
    gene: str,
    compartment: str,
    cmap: colors.Colormap,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
) -> None:
    indices = np.flatnonzero(included)
    positive = values[indices][values[indices] > 0]
    upper = float(np.quantile(positive, 0.99)) if len(positive) else 1.0
    relative = np.clip(values[indices] / max(upper, 1e-8), 0, 1)
    order = np.argsort(relative, kind="stable")
    indices = indices[order]
    relative = relative[order]
    ax.scatter(
        umap[indices, 0],
        umap[indices, 1],
        s=0.28,
        c=relative,
        cmap=cmap,
        vmin=0,
        vmax=1,
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
    ax.set_title(gene, fontsize=10, fontstyle="italic", weight="bold", pad=2)
    ax.text(
        0.5,
        0.995,
        compartment,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.2,
        color="#555555",
    )


def main() -> None:
    with h5py.File(H5AD, "r") as h5:
        cell_type = read_categorical(h5["obs"], "cell_type_paper9")
        umap = h5["obsm"]["X_umap"][:]
        expression = extract_expression(h5, list(FEATURES))

    included = np.isin(cell_type, list(CELL_MARKERS))
    x_limits = padded_limits(umap[included, 0])
    y_limits = padded_limits(umap[included, 1])
    feature_cmap = colors.LinearSegmentedColormap.from_list(
        "article_feature", ["#F3F3F3", "#F6B273", "#CF535E"]
    )

    fig = plt.figure(figsize=(16.8, 6.4), facecolor="white")
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=[3.2, 1.45],
        left=0.065,
        right=0.985,
        bottom=0.19,
        top=0.855,
        wspace=0.09,
    )
    dot_ax = fig.add_subplot(outer[0, 0])
    marker_mappable = make_dotplot(dot_ax)
    feature_grid = outer[0, 1].subgridspec(2, 2, wspace=0.035, hspace=0.11)
    feature_axes = [fig.add_subplot(feature_grid[row, col]) for row in range(2) for col in range(2)]

    for ax, (gene, compartment) in zip(feature_axes, FEATURES.items()):
        plot_feature(
            ax,
            umap,
            expression[gene],
            included,
            gene,
            compartment,
            feature_cmap,
            x_limits,
            y_limits,
        )

    marker_cax = fig.add_axes([0.405, 0.89, 0.12, 0.017])
    marker_colorbar = fig.colorbar(marker_mappable, cax=marker_cax, orientation="horizontal")
    marker_colorbar.set_ticks([-1, 0, 1, 2])
    marker_colorbar.ax.tick_params(labelsize=7, length=2)
    marker_colorbar.ax.set_title("Scaled average expression", fontsize=7.5, pad=3)
    marker_colorbar.outline.set_visible(False)

    feature_cax = fig.add_axes([0.775, 0.105, 0.14, 0.016])
    feature_colorbar = fig.colorbar(
        ScalarMappable(norm=colors.Normalize(0, 1), cmap=feature_cmap),
        cax=feature_cax,
        orientation="horizontal",
    )
    feature_colorbar.set_ticks([0, 0.5, 1])
    feature_colorbar.set_ticklabels(["Low", "", "High"])
    feature_colorbar.ax.tick_params(labelsize=7, length=0)
    feature_colorbar.ax.set_title("Relative expression (scaled per gene)", fontsize=7.5, pad=3)
    feature_colorbar.outline.set_visible(False)

    fig.text(0.018, 0.925, "B", fontsize=22, weight="normal", ha="left", va="top")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=400, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)
    print(f"Saved: {OUTPUT.with_suffix('.png')}")
    print(f"Saved: {OUTPUT.with_suffix('.pdf')}")
    print(f"Mapped paper9 cells: {int(included.sum()):,}")


if __name__ == "__main__":
    main()
