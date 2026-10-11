"""Build an epithelial Figure 3A in the visual structure of the reference paper.

The panel contains two views of the same epithelial-only UMAP: graph clusters
(C0-C8) and disease stage (Normal, OLK, OSCC). GSE181919 is used alone so a
cross-dataset batch effect cannot masquerade as the Normal-to-OSCC axis.
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
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import silhouette_score
import networkx as nx
import scanpy as sc


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
RESULT_DIR = ROOT / "results/paper9/figure3_epithelial"
FIGURE_DIR = ROOT / "figures/paper9/figure3_epithelial"
OUTPUT_STEM = FIGURE_DIR / "Fig3A_epithelial_clusters_reference_style"
DATASET = "A4_GSE181919"
STAGES = ["Normal", "OLK", "OSCC"]
SEED = 20260830
N_HVG = 3000
N_PCS = 15
TARGET_CLUSTERS = 9

# The reference uses the standard nine-color sequence and its first three
# colors for the clinical groups.
CLUSTER_COLORS = {
    "C0": "#1F77B4",
    "C1": "#FF7F0E",
    "C2": "#2CA02C",
    "C3": "#D62728",
    "C4": "#9467BD",
    "C5": "#8C564B",
    "C6": "#E377C2",
    "C7": "#BCBD22",
    "C8": "#17BECF",
}
STAGE_COLORS = {
    "Normal": "#1F77B4",
    "OLK": "#FF7F0E",
    "OSCC": "#2CA02C",
}


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


def load_epithelial() -> ad.AnnData:
    backed = ad.read_h5ad(H5AD, backed="r")
    cell_type = backed.obs["cell_type_paper9"].astype(str).to_numpy()
    dataset = backed.obs["dataset"].astype(str).to_numpy()
    stage = backed.obs["disease_status"].astype(str).to_numpy()
    keep = (
        (cell_type == "Epithelial cells")
        & (dataset == DATASET)
        & np.isin(stage, STAGES)
    )
    index = np.flatnonzero(keep)
    matrix = backed.X[index, :]
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    matrix = sparse.csr_matrix(matrix, dtype=np.float32)
    obs = backed.obs.iloc[index].copy()
    var = backed.var.copy()
    backed.file.close()
    if len(index) == 0:
        raise RuntimeError("No in-scope GSE181919 epithelial cells were found")
    out = ad.AnnData(X=matrix, obs=obs, var=var)
    out.obs_names_make_unique()
    for column in ("patient_id", "sample_id", "disease_status"):
        out.obs[column] = out.obs[column].astype(str).astype("category")
        out.obs[column] = out.obs[column].cat.remove_unused_categories()
    return out


def select_hvgs(adata: ad.AnnData) -> list[str]:
    patient_sizes = adata.obs["patient_id"].value_counts()
    eligible_patients = patient_sizes[patient_sizes >= 20].index
    hvg_source = adata[adata.obs["patient_id"].isin(eligible_patients)].copy()
    sc.pp.highly_variable_genes(
        hvg_source,
        n_top_genes=min(N_HVG + 500, adata.n_vars),
        flavor="cell_ranger",
        batch_key="patient_id",
        subset=False,
    )
    allowed = np.asarray(
        [not technical_gene(str(gene)) for gene in adata.var_names], dtype=bool
    )
    ranked = pd.DataFrame(
        {
            "gene": hvg_source.var_names.astype(str),
            "allowed": allowed,
            "nbatches": hvg_source.var.get(
                "highly_variable_nbatches", pd.Series(0, index=hvg_source.var_names)
            ).to_numpy(),
            "dispersion": hvg_source.var.get(
                "dispersions_norm", pd.Series(0.0, index=hvg_source.var_names)
            ).fillna(-np.inf).to_numpy(),
        }
    )
    ranked = ranked[ranked["allowed"]].sort_values(
        ["nbatches", "dispersion", "gene"],
        ascending=[False, False, True],
        kind="mergesort",
    )
    genes = ranked.head(min(N_HVG, len(ranked)))["gene"].tolist()
    if len(genes) < 500:
        raise RuntimeError(f"Only {len(genes)} usable highly variable genes were selected")
    return genes


def build_embedding(adata: ad.AnnData, hvg_genes: list[str]) -> ad.AnnData:
    work = adata[:, hvg_genes].copy()
    if sparse.issparse(work.X):
        work.X = sparse.csr_matrix(work.X).toarray().astype(np.float32, copy=False)
    else:
        work.X = np.asarray(work.X, dtype=np.float32)
    sc.pp.scale(work, max_value=10)
    n_comps = min(30, work.n_obs - 1, work.n_vars - 1)
    sc.tl.pca(work, n_comps=n_comps, svd_solver="randomized", random_state=SEED)
    sc.pp.neighbors(
        work,
        n_neighbors=20,
        n_pcs=min(N_PCS, n_comps),
        random_state=SEED,
    )
    sc.tl.umap(work, min_dist=0.35, spread=1.0, random_state=SEED)
    return work


def choose_nine_cluster_solution(work: ad.AnnData) -> tuple[pd.DataFrame, float, str]:
    coarse = np.round(np.arange(0.1, 1.01, 0.2), 2).tolist()
    fine = np.round(np.arange(0.1, 1.001, 0.05), 2).tolist()
    resolutions = coarse + [value for value in fine if value not in coarse]
    rng = np.random.default_rng(SEED)
    sample_n = min(2000, work.n_obs)
    sample_index = np.sort(rng.choice(work.n_obs, size=sample_n, replace=False))
    rows: list[dict[str, float | int | str]] = []
    label_keys: dict[float, str] = {}
    graph = nx.from_scipy_sparse_array(
        sparse.csr_matrix(work.obsp["connectivities"]), edge_attribute="weight"
    )
    for resolution in resolutions:
        key = f"louvain_r{resolution:.2f}".replace(".", "p")
        communities = nx.algorithms.community.louvain_communities(
            graph,
            weight="weight",
            resolution=float(resolution),
            seed=SEED,
        )
        labels = np.empty(work.n_obs, dtype=object)
        for cluster_id, members in enumerate(communities):
            labels[np.fromiter(members, dtype=int)] = str(cluster_id)
        work.obs[key] = pd.Categorical(labels)
        labels = np.asarray(labels, dtype=str)
        n_clusters = int(pd.Series(labels).nunique())
        sil = np.nan
        if 1 < n_clusters < sample_n:
            sil = float(
                silhouette_score(
                    work.obsm["X_pca"][sample_index, :N_PCS],
                    labels[sample_index],
                    metric="euclidean",
                )
            )
        rows.append(
            {
                "resolution": float(resolution),
                "n_clusters": n_clusters,
                "silhouette_sample_n": sample_n,
                "silhouette_pca15": sil,
                "original_reference_grid": resolution in coarse,
            }
        )
        label_keys[float(resolution)] = key

    audit = pd.DataFrame(rows).drop_duplicates("resolution").sort_values("resolution")
    exact = audit[audit["n_clusters"] == TARGET_CLUSTERS].copy()
    if exact.empty:
        closest = audit.assign(
            distance=(audit["n_clusters"] - TARGET_CLUSTERS).abs()
        ).sort_values(["distance", "silhouette_pca15"], ascending=[True, False])
        best = closest.iloc[0]
        raise RuntimeError(
            "No nine-cluster graph solution occurred in resolution 0.1-1.0; "
            f"closest was {int(best['n_clusters'])} clusters at {best['resolution']:.2f}."
        )
    exact["reference_grid_priority"] = exact["original_reference_grid"].astype(int)
    exact["distance_from_midpoint"] = (exact["resolution"] - 0.5).abs()
    chosen = exact.sort_values(
        ["reference_grid_priority", "silhouette_pca15", "distance_from_midpoint"],
        ascending=[False, False, True],
    ).iloc[0]
    resolution = float(chosen["resolution"])
    return audit, resolution, label_keys[resolution]


def assign_cluster_names(work: ad.AnnData, leiden_key: str) -> dict[str, str]:
    raw = work.obs[leiden_key].astype(str)
    sizes = raw.value_counts().rename_axis("leiden_cluster").reset_index(name="n_cells")
    sizes["numeric"] = pd.to_numeric(sizes["leiden_cluster"], errors="coerce")
    sizes = sizes.sort_values(
        ["n_cells", "numeric", "leiden_cluster"],
        ascending=[False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    mapping = {
        str(row.leiden_cluster): f"C{rank}"
        for rank, row in sizes.iterrows()
    }
    work.obs["epithelial_cluster"] = raw.map(mapping).astype("category")
    work.obs["epithelial_cluster"] = work.obs["epithelial_cluster"].cat.reorder_categories(
        list(CLUSTER_COLORS), ordered=True
    )
    return mapping


def robust_limits(embedding: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    x_low, x_high = np.quantile(embedding[:, 0], [0.001, 0.999])
    y_low, y_high = np.quantile(embedding[:, 1], [0.001, 0.999])
    x_pad = max((x_high - x_low) * 0.04, 0.1)
    y_pad = max((y_high - y_low) * 0.04, 0.1)
    return (x_low - x_pad, x_high + x_pad), (y_low - y_pad, y_high + y_pad)


def axis_glyph(ax: plt.Axes) -> None:
    color = "#222222"
    origin = (0.035, 0.055)
    ax.plot(
        [origin[0], origin[0] + 0.105],
        [origin[1], origin[1]],
        transform=ax.transAxes,
        color=color,
        lw=0.85,
        clip_on=False,
    )
    ax.plot(
        [origin[0], origin[0]],
        [origin[1], origin[1] + 0.125],
        transform=ax.transAxes,
        color=color,
        lw=0.85,
        clip_on=False,
    )
    ax.plot(
        [origin[0], origin[0] - 0.025],
        [origin[1], origin[1] - 0.018],
        transform=ax.transAxes,
        color=color,
        lw=0.85,
        clip_on=False,
    )
    ax.text(
        origin[0] + 0.055,
        origin[1] - 0.042,
        "X_umap1",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7.4,
    )
    ax.text(
        origin[0] - 0.027,
        origin[1] + 0.067,
        "X_umap2",
        transform=ax.transAxes,
        ha="right",
        va="center",
        rotation=90,
        fontsize=7.4,
    )


def scatter_groups(
    ax: plt.Axes,
    embedding: np.ndarray,
    labels: np.ndarray,
    palette: dict[str, str],
    draw_order: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
) -> None:
    for label, color in palette.items():
        selected = draw_order[labels[draw_order] == label]
        if len(selected) == 0:
            continue
        ax.scatter(
            embedding[selected, 0],
            embedding[selected, 1],
            s=3.0,
            c=color,
            alpha=0.78,
            linewidths=0,
            rasterized=True,
        )
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    axis_glyph(ax)


def legend_handles(palette: dict[str, str]) -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=5.2,
            markerfacecolor=color,
            markeredgecolor="none",
            label=label,
        )
        for label, color in palette.items()
    ]


def plot_reference_panel(work: ad.AnnData) -> None:
    embedding = work.obsm["X_umap"]
    clusters = work.obs["epithelial_cluster"].astype(str).to_numpy()
    stages = work.obs["disease_status"].astype(str).to_numpy()
    xlim, ylim = robust_limits(embedding)
    order = np.random.default_rng(SEED).permutation(work.n_obs)

    fig = plt.figure(figsize=(7.2, 3.15), facecolor="white")
    left = fig.add_axes([0.055, 0.08, 0.335, 0.84])
    right = fig.add_axes([0.505, 0.08, 0.335, 0.84])
    scatter_groups(left, embedding, clusters, CLUSTER_COLORS, order, xlim, ylim)
    scatter_groups(right, embedding, stages, STAGE_COLORS, order, xlim, ylim)

    left.legend(
        handles=legend_handles(CLUSTER_COLORS),
        loc="center left",
        bbox_to_anchor=(1.005, 0.58),
        frameon=False,
        fontsize=7.6,
        handlelength=0.7,
        handletextpad=0.35,
        labelspacing=0.38,
        borderaxespad=0,
    )
    right.legend(
        handles=legend_handles(STAGE_COLORS),
        loc="center left",
        bbox_to_anchor=(1.005, 0.62),
        frameon=False,
        fontsize=7.6,
        handlelength=0.7,
        handletextpad=0.35,
        labelspacing=0.38,
        borderaxespad=0,
    )
    fig.text(0.012, 0.965, "A", fontsize=23, fontweight="normal", va="top", ha="left")
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def export_results(
    work: ad.AnnData,
    hvg_genes: list[str],
    resolution_audit: pd.DataFrame,
    chosen_resolution: float,
    cluster_mapping: dict[str, str],
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    cells = work.obs[
        ["patient_id", "sample_id", "disease_status", "epithelial_cluster"]
    ].copy()
    cells.insert(0, "cell_id", work.obs_names.astype(str))
    cells["umap_1"] = work.obsm["X_umap"][:, 0]
    cells["umap_2"] = work.obsm["X_umap"][:, 1]
    cells.to_csv(
        RESULT_DIR / "Fig3A_epithelial_cells_and_umap.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )

    counts = (
        cells.groupby(["epithelial_cluster", "disease_status"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    totals = counts.groupby("disease_status", observed=False)["n_cells"].transform("sum")
    counts["fraction_within_stage"] = counts["n_cells"] / totals
    counts.to_csv(
        RESULT_DIR / "Fig3A_epithelial_cluster_stage_composition.csv",
        index=False,
        encoding="utf-8-sig",
    )
    resolution_audit.to_csv(
        RESULT_DIR / "Fig3A_epithelial_resolution_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame({"gene": hvg_genes}).to_csv(
        RESULT_DIR / "Fig3A_epithelial_hvg_genes.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "source_object": str(H5AD),
        "dataset": DATASET,
        "cell_type": "Epithelial cells",
        "included_stages": STAGES,
        "n_cells": int(work.n_obs),
        "n_genes_input": int(work.n_vars),
        "n_hvg": len(hvg_genes),
        "pca_components_for_graph": N_PCS,
        "batch_control": (
            "single GSE181919 cohort; patient-aware HVG selection; "
            "cross-dataset integration avoided"
        ),
        "clustering": "Louvain multi-resolution clustering on a 20-neighbor SNN graph",
        "chosen_resolution": chosen_resolution,
        "cluster_mapping_by_descending_size": cluster_mapping,
        "random_seed": SEED,
        "stage_cell_counts": {
            stage: int((work.obs["disease_status"].astype(str) == stage).sum())
            for stage in STAGES
        },
        "visual_reference": "s41746-025-02297-4 Figure 3A",
        "interpretation_boundary": (
            "Clusters are unsupervised transcriptional states and are not yet "
            "malignant subtypes; OLK has limited epithelial cell representation."
        ),
    }
    (RESULT_DIR / "Fig3A_epithelial_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    print("Loading GSE181919 epithelial cells...", flush=True)
    epithelial = load_epithelial()
    stage_counts = epithelial.obs["disease_status"].value_counts()
    print(
        f"  cells={epithelial.n_obs:,}; "
        + ", ".join(f"{stage}={int(stage_counts.get(stage, 0)):,}" for stage in STAGES),
        flush=True,
    )
    print("Selecting patient-aware HVGs and building epithelial embedding...", flush=True)
    hvg_genes = select_hvgs(epithelial)
    work = build_embedding(epithelial, hvg_genes)
    print("Auditing multi-resolution graph-clustering solutions...", flush=True)
    resolution_audit, chosen_resolution, leiden_key = choose_nine_cluster_solution(work)
    cluster_mapping = assign_cluster_names(work, leiden_key)
    print(
        f"  selected resolution={chosen_resolution:.2f}; "
        f"clusters={work.obs['epithelial_cluster'].nunique()}",
        flush=True,
    )
    export_results(
        work,
        hvg_genes,
        resolution_audit,
        chosen_resolution,
        cluster_mapping,
    )
    plot_reference_panel(work)
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}", flush=True)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
