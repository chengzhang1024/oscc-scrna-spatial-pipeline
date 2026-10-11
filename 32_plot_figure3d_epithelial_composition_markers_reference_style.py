"""Build Figure 3D: epithelial cluster tree, compositions, and marker heatmap.

The panel reuses Figure 3A C0-C8 labels and Figure 3C HCNV/LCNV calls. Cluster
markers and the hierarchy are computed from the same GSE181919 epithelial
expression matrix. Composition bars are descriptive cell fractions.
"""

from __future__ import annotations

import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache"
TASK_TEMP = CACHE_ROOT / "codex_tmp"
NUMBA_CACHE = CACHE_ROOT / "numba_cache"
TASK_TEMP.mkdir(parents=True, exist_ok=True)
NUMBA_CACHE.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TASK_TEMP)
os.environ["TMP"] = str(TASK_TEMP)
os.environ["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.cluster.hierarchy import dendrogram, linkage, set_link_color_palette
from scipy.spatial.distance import pdist


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

H5AD = ROOT / "results/integrated_3ds_annotated_paper9.h5ad"
FIG3A_CELLS = ROOT / "results/paper9/figure3_epithelial/Fig3A_epithelial_cells_and_umap.csv.gz"
FIG3C_CELLS = ROOT / "results/paper9/figure3_epithelial/Fig3C_epithelial_cnv_groups_and_umap.csv.gz"
RESULT_DIR = ROOT / "results/paper9/figure3_epithelial"
FIGURE_DIR = ROOT / "figures/paper9/figure3_epithelial"
OUTPUT_STEM = FIGURE_DIR / "Fig3D_epithelial_composition_markers_reference_style"
CLUSTERS = [f"C{i}" for i in range(9)]
STAGES = ["Normal", "OLK", "OSCC"]
CNV_GROUPS = ["LCNV", "HCNV"]
TOP_MARKERS_PER_CLUSTER = 50
TOP_LABEL_MARKERS = 5

STAGE_COLORS = {
    "Normal": "#247BA0",
    "OLK": "#1B936A",
    "OSCC": "#D52B75",
}
CNV_COLORS = {
    "HCNV": "#DF2727",
    "LCNV": "#397EAC",
}
TREE_COLORS = ["#E31A1C", "#1954A6"]
TREE_TRUNK = "#59B527"

QC_FLAGS = [
    {
        "epithelial_cluster": "C5",
        "qc_status": "review_non_epithelial_mixed",
        "evidence": "DLK1/APOC1/SPARCL1/NNMT mesenchymal-muscle-like program",
        "recommended_use": "display transparently; do not assign a definitive epithelial state",
    },
    {
        "epithelial_cluster": "C8",
        "qc_status": "qc_excluded_immune_like",
        "evidence": "CD52/CD3D/TRAC/PTPRC/CD3E immune program",
        "recommended_use": "exclude before biological epithelial subtype interpretation",
    },
]


def technical_gene(gene: str) -> bool:
    return (
        gene.startswith("MT-")
        or gene.startswith("RPL")
        or gene.startswith("RPS")
        or gene.startswith("IGH")
        or gene.startswith("IGK")
        or gene.startswith("IGL")
        or gene.startswith("TRAV")
        or gene.startswith("TRBV")
    )


def load_inputs() -> tuple[pd.DataFrame, sparse.csr_matrix, np.ndarray]:
    cells = pd.read_csv(FIG3A_CELLS)
    cnv = pd.read_csv(FIG3C_CELLS, usecols=["cell_id", "cnv_group"])
    cells = cells.merge(cnv, on="cell_id", how="left", validate="one_to_one")
    if len(cells) == 0 or cells["cnv_group"].isna().any():
        raise RuntimeError("Figure 3A/3C cell tables do not align")
    if set(cells["epithelial_cluster"].astype(str)) != set(CLUSTERS):
        raise RuntimeError("Figure 3A does not contain exactly C0-C8")

    backed = ad.read_h5ad(H5AD, backed="r")
    lookup = pd.Series(np.arange(backed.n_obs), index=backed.obs_names.astype(str))
    row_index = lookup.reindex(cells["cell_id"].astype(str)).to_numpy()
    if np.isnan(row_index).any():
        backed.file.close()
        raise RuntimeError("Some Figure 3A cells are absent from the source object")
    row_index = row_index.astype(int)
    matrix = backed.X[row_index, :]
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    genes = backed.var_names.astype(str).to_numpy()
    backed.file.close()
    return cells, matrix, genes


def calculate_markers(
    cells: pd.DataFrame,
    matrix: sparse.csr_matrix,
    genes: np.ndarray,
) -> tuple[pd.DataFrame, list[str], np.ndarray, np.ndarray]:
    labels = cells["epithelial_cluster"].astype(str).to_numpy()
    allowed = np.asarray([not technical_gene(gene) for gene in genes], dtype=bool)
    total_sum = np.asarray(matrix.sum(axis=0)).ravel()
    total_nnz = np.asarray(matrix.getnnz(axis=0)).ravel()
    marker_rows: list[dict[str, float | int | str]] = []
    selected_by_cluster: dict[str, list[str]] = {}
    mean_by_cluster = np.empty((len(CLUSTERS), len(genes)), dtype=np.float32)

    for cluster_index, cluster in enumerate(CLUSTERS):
        mask = labels == cluster
        n_in = int(mask.sum())
        n_out = len(labels) - n_in
        sub = matrix[mask]
        sum_in = np.asarray(sub.sum(axis=0)).ravel()
        nnz_in = np.asarray(sub.getnnz(axis=0)).ravel()
        mean_in = sum_in / max(n_in, 1)
        mean_out = (total_sum - sum_in) / max(n_out, 1)
        pct_in = nnz_in / max(n_in, 1)
        pct_out = (total_nnz - nnz_in) / max(n_out, 1)
        score = (mean_in - mean_out) * np.maximum(pct_in - pct_out, 0.0)
        valid_score = np.where(allowed & (pct_in >= 0.10), score, -np.inf)
        order = np.argsort(valid_score, kind="mergesort")[::-1]
        order = order[np.isfinite(valid_score[order])][:TOP_MARKERS_PER_CLUSTER]
        selected_by_cluster[cluster] = genes[order].tolist()
        mean_by_cluster[cluster_index] = mean_in.astype(np.float32)
        for rank, gene_index in enumerate(order, start=1):
            marker_rows.append(
                {
                    "epithelial_cluster": cluster,
                    "rank": rank,
                    "gene": genes[gene_index],
                    "marker_score": float(score[gene_index]),
                    "mean_in": float(mean_in[gene_index]),
                    "mean_out": float(mean_out[gene_index]),
                    "pct_in": float(pct_in[gene_index]),
                    "pct_out": float(pct_out[gene_index]),
                    "mean_difference": float(mean_in[gene_index] - mean_out[gene_index]),
                }
            )

    heatmap_genes: list[str] = []
    seen: set[str] = set()
    gene_blocks: list[str] = []
    for cluster in CLUSTERS:
        for gene in selected_by_cluster[cluster]:
            if gene not in seen:
                seen.add(gene)
                heatmap_genes.append(gene)
                gene_blocks.append(cluster)
    gene_lookup = pd.Series(np.arange(len(genes)), index=genes)
    heat_indices = gene_lookup.reindex(heatmap_genes).to_numpy(dtype=int)
    heat_means = mean_by_cluster[:, heat_indices]
    gene_mean = heat_means.mean(axis=0, keepdims=True)
    gene_sd = heat_means.std(axis=0, ddof=0, keepdims=True)
    heat_z = (heat_means - gene_mean) / (gene_sd + 1e-6)
    heat_z = np.clip(heat_z, -2.0, 2.0).astype(np.float32)
    return pd.DataFrame(marker_rows), heatmap_genes, np.asarray(gene_blocks), heat_z


def cluster_hierarchy(heat_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = pdist(heat_z, metric="correlation")
    distance = np.nan_to_num(distance, nan=0.0, posinf=1.0, neginf=1.0)
    tree = linkage(distance, method="average", optimal_ordering=True)
    preview = dendrogram(tree, no_plot=True)
    return tree, np.asarray(preview["leaves"], dtype=int)


def composition_table(
    cells: pd.DataFrame,
    column: str,
    levels: list[str],
) -> pd.DataFrame:
    table = (
        cells.groupby(["epithelial_cluster", column], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    complete = pd.MultiIndex.from_product(
        [CLUSTERS, levels], names=["epithelial_cluster", column]
    ).to_frame(index=False)
    table = complete.merge(table, on=["epithelial_cluster", column], how="left")
    table["n_cells"] = table["n_cells"].fillna(0).astype(int)
    table["fraction_within_cluster"] = table["n_cells"] / table.groupby(
        "epithelial_cluster", observed=False
    )["n_cells"].transform("sum")
    return table


def add_header_strip(ax: plt.Axes, text: str) -> None:
    ax.add_patch(
        Rectangle(
            (0, 1.04),
            1,
            0.085,
            transform=ax.transAxes,
            facecolor="#E7E7E7",
            edgecolor="#555555",
            linewidth=0.65,
            clip_on=False,
        )
    )
    ax.text(
        0.5,
        1.082,
        text,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.8,
        fontstyle="italic",
    )


def stacked_bars(
    ax: plt.Axes,
    table: pd.DataFrame,
    column: str,
    levels: list[str],
    colors: dict[str, str],
    leaf_clusters: list[str],
) -> None:
    y_positions = np.arange(len(leaf_clusters)) * 10 + 5
    for y, cluster in zip(y_positions, leaf_clusters):
        left = 0.0
        subset = table[table["epithelial_cluster"] == cluster].set_index(column)
        for level in levels:
            width = float(subset.loc[level, "fraction_within_cluster"] * 100)
            ax.barh(
                y,
                width,
                left=left,
                height=8.1,
                color=colors[level],
                edgecolor="none",
                align="center",
            )
            left += width
    ax.set_xlim(0, 100)
    ax.set_ylim(0, len(leaf_clusters) * 10)
    ax.invert_yaxis()
    ax.set_yticks([])
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.tick_params(axis="x", labelsize=5.7, width=0.6, length=2.5, pad=1.5)
    for side in ("top", "left", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.7)


def plot_figure(
    cells: pd.DataFrame,
    markers: pd.DataFrame,
    heatmap_genes: list[str],
    heat_z: np.ndarray,
    tree: np.ndarray,
    stage_comp: pd.DataFrame,
    cnv_comp: pd.DataFrame,
) -> list[str]:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.2, 3.8), facecolor="white")
    ax_tree = fig.add_axes([0.080, 0.135, 0.205, 0.680])
    ax_stage = fig.add_axes([0.305, 0.135, 0.120, 0.680])
    ax_cnv = fig.add_axes([0.440, 0.135, 0.120, 0.680])
    ax_heat = fig.add_axes([0.575, 0.135, 0.245, 0.680])
    ax_labels = fig.add_axes([0.825, 0.135, 0.170, 0.680])

    set_link_color_palette(TREE_COLORS)
    color_threshold = float(tree[:, 2].max() * 0.62)
    tree_plot = dendrogram(
        tree,
        orientation="right",
        no_labels=True,
        color_threshold=color_threshold,
        above_threshold_color=TREE_TRUNK,
        ax=ax_tree,
    )
    leaf_indices = [int(value) for value in tree_plot["leaves"]]
    leaf_clusters = [CLUSTERS[index] for index in leaf_indices]
    ax_tree.invert_yaxis()
    ax_tree.set_yticks([])
    for side in ("top", "left", "right"):
        ax_tree.spines[side].set_visible(False)
    ax_tree.spines["bottom"].set_linewidth(0.7)
    ax_tree.tick_params(axis="x", labelsize=5.7, width=0.6, length=2.5, pad=1.5)
    xmax = ax_tree.get_xlim()[1]
    ax_tree.set_xlim(0, xmax * 1.10)
    leaf_colors = tree_plot.get("leaves_color_list", ["#222222"] * len(leaf_clusters))
    for position, (cluster, color) in enumerate(zip(leaf_clusters, leaf_colors)):
        ax_tree.text(
            xmax * 1.025,
            position * 10 + 5,
            cluster.replace("C", ""),
            ha="left",
            va="center",
            fontsize=7.3,
            fontweight="bold",
            color=color,
            clip_on=False,
        )

    stacked_bars(ax_stage, stage_comp, "disease_status", STAGES, STAGE_COLORS, leaf_clusters)
    stacked_bars(ax_cnv, cnv_comp, "cnv_group", CNV_GROUPS, CNV_COLORS, leaf_clusters)
    add_header_strip(ax_stage, "Composition by group")
    add_header_strip(ax_cnv, "Composition by CNV_group")

    ordered_heat = heat_z[leaf_indices]
    ax_heat.imshow(
        ordered_heat,
        aspect="auto",
        interpolation="nearest",
        cmap="RdBu_r",
        vmin=-2,
        vmax=2,
        origin="upper",
        rasterized=True,
    )
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    for spine in ax_heat.spines.values():
        spine.set_visible(False)

    top_markers = (
        markers[markers["rank"] <= TOP_LABEL_MARKERS]
        .sort_values(["epithelial_cluster", "rank"])
        .groupby("epithelial_cluster", observed=False)["gene"]
        .apply(list)
        .to_dict()
    )
    ax_labels.set_xlim(0, 1)
    ax_labels.set_ylim(0, len(leaf_clusters))
    ax_labels.invert_yaxis()
    ax_labels.axis("off")
    for row, cluster in enumerate(leaf_clusters):
        text = ", ".join(top_markers.get(cluster, []))
        ax_labels.text(0.0, row + 0.5, text, ha="left", va="center", fontsize=6.6)

    gradient_axis = fig.add_axes([0.680, 0.865, 0.065, 0.025])
    gradient = np.linspace(2, -2, 256, dtype=float)[None, :]
    gradient_axis.imshow(gradient, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    gradient_axis.set_xticks([0, 64, 128, 191, 255])
    gradient_axis.set_xticklabels(["2", "1", "0", "-1", "-2"], fontsize=5.5)
    gradient_axis.xaxis.set_ticks_position("top")
    gradient_axis.tick_params(axis="x", length=1.8, width=0.5, pad=0.5)
    gradient_axis.set_yticks([])
    for spine in gradient_axis.spines.values():
        spine.set_visible(False)
    fig.text(0.7125, 0.835, "Z-score", fontsize=6.5, fontweight="bold", ha="center")

    group_handles = [
        Line2D([0], [0], color=color, lw=3.2, label=f"cluster{i + 1}")
        for i, color in enumerate(["#E31A1C", "#1954A6", TREE_TRUNK])
    ]
    cnv_handles = [Patch(facecolor=CNV_COLORS[group], edgecolor="none", label=group) for group in ["HCNV", "LCNV"]]
    stage_handles = [Patch(facecolor=STAGE_COLORS[stage], edgecolor="none", label=stage) for stage in STAGES]
    legend_group = fig.legend(
        handles=group_handles,
        title="group",
        loc="upper left",
        bbox_to_anchor=(0.080, 0.935),
        frameon=False,
        fontsize=5.7,
        title_fontsize=6.0,
        handlelength=2.0,
        handletextpad=0.35,
        labelspacing=0.35,
    )
    fig.add_artist(legend_group)
    legend_cnv = fig.legend(
        handles=cnv_handles,
        title="CNV group",
        loc="upper left",
        bbox_to_anchor=(0.160, 0.935),
        frameon=False,
        fontsize=5.7,
        title_fontsize=6.0,
        handlelength=0.9,
        handletextpad=0.30,
        labelspacing=0.35,
    )
    fig.add_artist(legend_cnv)
    fig.legend(
        handles=stage_handles,
        title="Group",
        loc="upper left",
        bbox_to_anchor=(0.225, 0.935),
        frameon=False,
        fontsize=5.7,
        title_fontsize=6.0,
        handlelength=0.9,
        handletextpad=0.30,
        labelspacing=0.35,
    )
    fig.text(0.010, 0.970, "D", fontsize=23, fontweight="normal", va="top", ha="left")

    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    return leaf_clusters


def export_results(
    cells: pd.DataFrame,
    markers: pd.DataFrame,
    heatmap_genes: list[str],
    heat_z: np.ndarray,
    tree: np.ndarray,
    leaf_clusters: list[str],
    stage_comp: pd.DataFrame,
    cnv_comp: pd.DataFrame,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    markers.to_csv(
        RESULT_DIR / "Fig3D_epithelial_cluster_markers.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    stage_comp.to_csv(
        RESULT_DIR / "Fig3D_epithelial_composition_by_stage.csv",
        index=False,
        encoding="utf-8-sig",
    )
    cnv_comp.to_csv(
        RESULT_DIR / "Fig3D_epithelial_composition_by_cnv_group.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "heatmap_column": np.arange(len(heatmap_genes)),
            "gene": heatmap_genes,
        }
    ).to_csv(
        RESULT_DIR / "Fig3D_epithelial_marker_heatmap_gene_order.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        heat_z,
        index=CLUSTERS,
        columns=heatmap_genes,
    ).rename_axis("epithelial_cluster").to_csv(
        RESULT_DIR / "Fig3D_epithelial_marker_heatmap_zscores.csv.gz",
        compression="gzip",
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {
            "display_order_top_to_bottom": np.arange(len(leaf_clusters)),
            "epithelial_cluster": leaf_clusters,
        }
    ).to_csv(
        RESULT_DIR / "Fig3D_epithelial_dendrogram_order.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.save(RESULT_DIR / "Fig3D_epithelial_linkage_matrix.npy", tree)
    pd.DataFrame(QC_FLAGS).to_csv(
        RESULT_DIR / "Fig3D_epithelial_cluster_qc_flags.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "source_object": str(H5AD),
        "figure3a_cell_table": str(FIG3A_CELLS),
        "figure3c_cell_table": str(FIG3C_CELLS),
        "n_cells": int(len(cells)),
        "clusters": CLUSTERS,
        "display_order_top_to_bottom": leaf_clusters,
        "hierarchy": "average linkage of correlation distances among cluster marker-expression z-scores",
        "marker_score": "(mean_in - mean_out) * max(pct_in - pct_out, 0)",
        "marker_minimum_detection_fraction": 0.10,
        "top_markers_per_cluster_for_heatmap": TOP_MARKERS_PER_CLUSTER,
        "top_markers_per_cluster_for_labels": TOP_LABEL_MARKERS,
        "heatmap_gene_count": len(heatmap_genes),
        "heatmap_zscore_axis": "per gene across C0-C8, clipped to [-2, 2]",
        "stage_composition_scope": "cell fractions within each epithelial cluster",
        "cnv_composition_scope": "cell fractions within each epithelial cluster",
        "visual_reference": "s41746-025-02297-4 Figure 3D",
        "cluster_qc_flags": QC_FLAGS,
        "interpretation_boundary": (
            "Composition bars are descriptive cell-level fractions and are not "
            "patient-balanced inferential tests. Marker labels require biological "
            "annotation review before assigning functional subtype names."
        ),
    }
    (RESULT_DIR / "Fig3D_epithelial_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    print("Loading Figure 3A/3C cells and epithelial expression...", flush=True)
    cells, matrix, genes = load_inputs()
    print(f"  cells={len(cells):,}; genes={len(genes):,}", flush=True)
    print("Calculating C0-C8 marker programs...", flush=True)
    markers, heatmap_genes, _, heat_z = calculate_markers(cells, matrix, genes)
    print(f"  selected heatmap genes={len(heatmap_genes):,}", flush=True)
    tree, _ = cluster_hierarchy(heat_z)
    stage_comp = composition_table(cells, "disease_status", STAGES)
    cnv_comp = composition_table(cells, "cnv_group", CNV_GROUPS)
    leaf_clusters = plot_figure(
        cells,
        markers,
        heatmap_genes,
        heat_z,
        tree,
        stage_comp,
        cnv_comp,
    )
    export_results(
        cells,
        markers,
        heatmap_genes,
        heat_z,
        tree,
        leaf_clusters,
        stage_comp,
        cnv_comp,
    )
    print(f"  display order={','.join(leaf_clusters)}", flush=True)
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}", flush=True)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
