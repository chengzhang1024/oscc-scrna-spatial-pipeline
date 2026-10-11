"""Build Figure 3B: reference-style epithelial expression-CNV heatmap.

The analysis uses GSE181919 endothelial cells as the diploid reference and
the exact epithelial cells/clusters from Figure 3A. Genes are ordered by the
GENCODE v50 GRCh38 annotation and smoothed within each autosome.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
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
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.ndimage import uniform_filter1d
from scipy.stats import kruskal


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
GTF = ROOT / "data/reference_annotation/gencode_v50/gencode.v50.annotation.gtf.gz"
RESULT_DIR = ROOT / "results/paper9/figure3_epithelial"
FIGURE_DIR = ROOT / "figures/paper9/figure3_epithelial"
OUTPUT_STEM = FIGURE_DIR / "Fig3B_epithelial_infercnv_reference_style"
DATASET = "A4_GSE181919"
AUTOSOMES = [f"chr{index}" for index in range(1, 23)]
SEED = 20260831
WINDOW_SIZE = 101
MIN_MEAN_EXPRESSION = 0.10
NOISE_MULTIPLIER = 1.50
CHUNK_SIZE = 192
PLOT_EPI_CELLS = 1200
PLOT_REF_CELLS = 80
VMIN = -0.20
VMAX = 0.40

CNV_CMAP = LinearSegmentedColormap.from_list(
    "reference_cnv", [(0.0, "#0000F2"), (1 / 3, "#FFFFFF"), (1.0, "#FF1515")]
)
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_gene_name(attributes: str) -> str | None:
    match = re.search(r'(?:^|;\s*)gene_name "([^"]+)"', attributes)
    return match.group(1) if match else None


def parse_gtf_gene_order() -> pd.DataFrame:
    if not GTF.exists() or GTF.stat().st_size == 0:
        raise FileNotFoundError(f"GENCODE annotation is missing: {GTF}")
    rows: list[tuple[str, str, int, int]] = []
    with gzip.open(GTF, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 9 or fields[2] != "gene" or fields[0] not in AUTOSOMES:
                continue
            gene = parse_gene_name(fields[8])
            if gene:
                rows.append((gene, fields[0], int(fields[3]), int(fields[4])))
    annotation = pd.DataFrame(rows, columns=["gene", "chromosome", "start", "end"])
    annotation["chromosome_index"] = annotation["chromosome"].str[3:].astype(int)
    annotation = annotation.sort_values(
        ["chromosome_index", "start", "end", "gene"], kind="mergesort"
    )
    # Symbols duplicated across loci are resolved to the first primary-chromosome
    # occurrence, making the ordering deterministic and directly auditable.
    annotation = annotation.drop_duplicates("gene", keep="first").reset_index(drop=True)
    return annotation


def load_indices() -> tuple[ad.AnnData, pd.DataFrame, np.ndarray, np.ndarray]:
    fig3a = pd.read_csv(FIG3A_CELLS)
    required = {"cell_id", "epithelial_cluster", "disease_status"}
    if not required.issubset(fig3a.columns):
        raise RuntimeError(f"Figure 3A cell table lacks {sorted(required - set(fig3a.columns))}")

    backed = ad.read_h5ad(H5AD, backed="r")
    name_to_index = pd.Series(np.arange(backed.n_obs), index=backed.obs_names.astype(str))
    epi_index = name_to_index.reindex(fig3a["cell_id"].astype(str)).to_numpy()
    if np.isnan(epi_index).any():
        backed.file.close()
        raise RuntimeError("Some Figure 3A epithelial cell IDs are absent from the source object")
    epi_index = epi_index.astype(int)

    dataset = backed.obs["dataset"].astype(str).to_numpy()
    cell_type = backed.obs["cell_type_paper9"].astype(str).to_numpy()
    ref_index = np.flatnonzero((dataset == DATASET) & (cell_type == "Endothelial cells"))
    if len(ref_index) == 0:
        backed.file.close()
        raise RuntimeError("No GSE181919 endothelial reference cells were found")
    return backed, fig3a, epi_index, ref_index


def mean_expression(matrix: object, indices: np.ndarray) -> np.ndarray:
    selected = matrix[indices, :]
    if hasattr(selected, "to_memory"):
        selected = selected.to_memory()
    if sparse.issparse(selected):
        return np.asarray(selected.mean(axis=0)).ravel().astype(np.float32)
    return np.asarray(selected, dtype=np.float32).mean(axis=0)


def extract_dense(matrix: object, rows: np.ndarray, columns: np.ndarray) -> np.ndarray:
    selected = matrix[rows, :]
    if hasattr(selected, "to_memory"):
        selected = selected.to_memory()
    selected = selected[:, columns]
    if sparse.issparse(selected):
        selected = selected.toarray()
    return np.asarray(selected, dtype=np.float32)


def ordered_expression_genes(
    backed: ad.AnnData,
    annotation: pd.DataFrame,
    epi_index: np.ndarray,
    ref_index: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    var_names = backed.var_names.astype(str)
    gene_to_index = pd.Series(np.arange(backed.n_vars), index=var_names)
    matched = annotation[annotation["gene"].isin(gene_to_index.index)].copy()
    matched["var_index"] = gene_to_index.reindex(matched["gene"]).to_numpy(dtype=int)
    epi_mean = mean_expression(backed.X, epi_index)
    ref_mean_all = mean_expression(backed.X, ref_index)
    combined_mean = (
        epi_mean * len(epi_index) + ref_mean_all * len(ref_index)
    ) / (len(epi_index) + len(ref_index))
    matched["combined_mean_expression"] = combined_mean[matched["var_index"].to_numpy()]
    matched = matched[matched["combined_mean_expression"] >= MIN_MEAN_EXPRESSION].copy()
    matched = matched.sort_values(
        ["chromosome_index", "start", "end", "gene"], kind="mergesort"
    ).reset_index(drop=True)
    if len(matched) < 3000:
        raise RuntimeError(f"Only {len(matched)} ordered expressed autosomal genes remain")
    gene_indices = matched["var_index"].to_numpy(dtype=int)
    ref_gene_mean = ref_mean_all[gene_indices].astype(np.float32)
    return matched, gene_indices, ref_gene_mean


def chromosome_slices(genes: pd.DataFrame) -> list[tuple[str, int, int]]:
    result: list[tuple[str, int, int]] = []
    chromosomes = genes["chromosome"].to_numpy()
    for chromosome in AUTOSOMES:
        locations = np.flatnonzero(chromosomes == chromosome)
        if len(locations):
            result.append((chromosome, int(locations[0]), int(locations[-1] + 1)))
    return result


def smooth_relative_expression(
    dense: np.ndarray,
    reference_mean: np.ndarray,
    slices: list[tuple[str, int, int]],
) -> np.ndarray:
    relative = np.clip(dense - reference_mean[None, :], -3.0, 3.0).astype(
        np.float32, copy=False
    )
    smoothed = np.empty_like(relative)
    for _, start, end in slices:
        width = min(WINDOW_SIZE, end - start)
        if width % 2 == 0:
            width -= 1
        width = max(width, 1)
        smoothed[:, start:end] = uniform_filter1d(
            relative[:, start:end], size=width, axis=1, mode="nearest"
        )
    smoothed -= np.median(smoothed, axis=1, keepdims=True)
    return smoothed


def process_rows(
    matrix: object,
    rows: np.ndarray,
    gene_indices: np.ndarray,
    reference_mean: np.ndarray,
    slices: list[tuple[str, int, int]],
    noise_floor: np.ndarray | None = None,
    retain: set[int] | None = None,
) -> tuple[pd.DataFrame, dict[int, np.ndarray], np.ndarray, np.ndarray]:
    score_rows: list[dict[str, float | int]] = []
    retained: dict[int, np.ndarray] = {}
    running_sum = np.zeros(len(gene_indices), dtype=np.float64)
    running_sq = np.zeros(len(gene_indices), dtype=np.float64)
    running_n = 0
    for offset in range(0, len(rows), CHUNK_SIZE):
        chunk_rows = rows[offset : offset + CHUNK_SIZE]
        dense = extract_dense(matrix, chunk_rows, gene_indices)
        values = smooth_relative_expression(dense, reference_mean, slices)
        if noise_floor is not None:
            values = np.sign(values) * np.maximum(np.abs(values) - noise_floor[None, :], 0.0)
            values -= np.median(values, axis=1, keepdims=True)
        running_sum += values.sum(axis=0, dtype=np.float64)
        running_sq += np.square(values, dtype=np.float32).sum(axis=0, dtype=np.float64)
        running_n += len(chunk_rows)
        scores = np.sqrt(np.mean(np.square(values, dtype=np.float32), axis=1))
        for local, (global_row, score) in enumerate(zip(chunk_rows, scores)):
            score_rows.append({"source_row": int(global_row), "cnv_score": float(score)})
            if retain is not None and int(global_row) in retain:
                retained[int(global_row)] = values[local].astype(np.float32, copy=True)
    mean = running_sum / max(running_n, 1)
    variance = np.maximum(running_sq / max(running_n, 1) - mean**2, 0.0)
    return pd.DataFrame(score_rows), retained, mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def stratified_sample(fig3a: pd.DataFrame, epi_index: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    table = fig3a.copy()
    table["source_row"] = epi_index
    cluster_sizes = table["epithelial_cluster"].value_counts().sort_index()
    raw_target = cluster_sizes / cluster_sizes.sum() * PLOT_EPI_CELLS
    target = np.floor(raw_target).astype(int)
    target = np.maximum(target, 25)
    while target.sum() > PLOT_EPI_CELLS:
        eligible = target[target > 25]
        target.loc[eligible.idxmax()] -= 1
    while target.sum() < PLOT_EPI_CELLS:
        residual = raw_target - target
        target.loc[residual.idxmax()] += 1
    selected: list[int] = []
    for cluster in sorted(cluster_sizes.index):
        candidates = table.loc[table["epithelial_cluster"] == cluster, "source_row"].to_numpy(dtype=int)
        n = min(int(target.loc[cluster]), len(candidates))
        selected.extend(rng.choice(candidates, size=n, replace=False).tolist())
    return np.asarray(selected, dtype=int)


def order_heatmap_rows(
    selected_rows: np.ndarray,
    retained: dict[int, np.ndarray],
    fig3a: pd.DataFrame,
    epi_index: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    labels = pd.DataFrame(
        {
            "source_row": epi_index,
            "cell_id": fig3a["cell_id"].astype(str),
            "epithelial_cluster": fig3a["epithelial_cluster"].astype(str),
        }
    ).set_index("source_row")
    records = []
    matrices = []
    for cluster in [f"C{i}" for i in range(9)]:
        cluster_rows = [row for row in selected_rows if labels.loc[row, "epithelial_cluster"] == cluster]
        if not cluster_rows:
            continue
        block = np.vstack([retained[row] for row in cluster_rows])
        # A weighted genome-wide index provides a deterministic within-cluster
        # order while preserving the C0-C8 group sequence used by Figure 3A.
        weights = np.linspace(-1.0, 1.0, block.shape[1], dtype=np.float32)
        order = np.argsort(block @ weights, kind="mergesort")
        matrices.append(block[order])
        for display_row, local in enumerate(order):
            source_row = cluster_rows[int(local)]
            records.append(
                {
                    "display_row": len(records),
                    "source_row": source_row,
                    "cell_id": labels.loc[source_row, "cell_id"],
                    "epithelial_cluster": cluster,
                }
            )
    return np.vstack(matrices), pd.DataFrame(records)


def plot_figure(
    epithelial_matrix: np.ndarray,
    reference_matrix: np.ndarray,
    genes: pd.DataFrame,
    slices: list[tuple[str, int, int]],
    cell_scores: pd.DataFrame,
    overall_p: float,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.25, 5.05), facecolor="white")
    heat_left = 0.115
    heat_right = 0.845
    epi_bottom = 0.425
    ref_bottom = 0.375
    ref_height = 0.038
    epi_height = 0.445
    ax_epi = fig.add_axes([heat_left, epi_bottom, heat_right - heat_left, epi_height])
    ax_ref = fig.add_axes([heat_left, ref_bottom, heat_right - heat_left, ref_height])

    ax_epi.imshow(
        epithelial_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=CNV_CMAP,
        vmin=VMIN,
        vmax=VMAX,
        rasterized=True,
    )
    ax_ref.imshow(
        reference_matrix,
        aspect="auto",
        interpolation="nearest",
        cmap=CNV_CMAP,
        vmin=VMIN,
        vmax=VMAX,
        rasterized=True,
    )
    for axis in (ax_epi, ax_ref):
        axis.set_xticks([])
        axis.set_yticks([])
        for spine in axis.spines.values():
            spine.set_visible(True)
            spine.set_color("black")
            spine.set_linewidth(1.35)
        for _, start, _ in slices[1:]:
            axis.axvline(start - 0.5, color="#0068A8", lw=0.75, alpha=0.95)

    n_genes = len(genes)
    label_positions = np.asarray(
        [
            heat_left
            + (((start + end - 1) / 2) / max(n_genes - 1, 1)) * (heat_right - heat_left)
            for _, start, end in slices
        ],
        dtype=float,
    )
    minimum_label_spacing = 0.024
    for index in range(1, len(label_positions)):
        label_positions[index] = max(
            label_positions[index], label_positions[index - 1] + minimum_label_spacing
        )
    if label_positions[-1] > heat_right:
        label_positions -= label_positions[-1] - heat_right
    for index in range(len(label_positions) - 2, -1, -1):
        label_positions[index] = min(
            label_positions[index], label_positions[index + 1] - minimum_label_spacing
        )
    if label_positions[0] < heat_left:
        label_positions += heat_left - label_positions[0]

    for (chromosome, start, end), x_fig in zip(slices, label_positions):
        center = (start + end - 1) / 2
        fig.text(
            x_fig,
            0.900,
            chromosome.replace("chr", "Chr"),
            ha="center",
            va="bottom",
            fontsize=4.4,
            fontstyle="italic",
        )
        start_fig = heat_left + (start / n_genes) * (heat_right - heat_left)
        end_fig = heat_left + (end / n_genes) * (heat_right - heat_left)
        fig.add_artist(
            Line2D(
                [start_fig + 0.0007, end_fig - 0.0007],
                [0.886, 0.886],
                transform=fig.transFigure,
                color="black",
                lw=2.0,
                solid_capstyle="butt",
            )
        )

    fig.text(0.083, epi_bottom + epi_height / 2, "Epi", rotation=90, ha="center", va="center", fontsize=10)
    fig.text(0.083, ref_bottom + ref_height / 2, "Ref.", rotation=90, ha="center", va="center", fontsize=9)
    fig.text(0.018, 0.970, "B", fontsize=27, fontweight="normal", va="top", ha="left")

    color_axis = fig.add_axes([0.868, ref_bottom, 0.025, epi_bottom + epi_height - ref_bottom])
    colorbar = fig.colorbar(
        ScalarMappable(norm=Normalize(vmin=VMIN, vmax=VMAX), cmap=CNV_CMAP),
        cax=color_axis,
    )
    colorbar.set_ticks([VMIN, 0.0, VMAX])
    colorbar.set_ticklabels([f"{VMIN:.1f}", "", f"{VMAX:.1f}"])
    colorbar.ax.tick_params(labelsize=9, width=0.9, length=3.5)
    for label in colorbar.ax.get_yticklabels():
        label.set_rotation(90)
        label.set_va("center")
        label.set_ha("center")
    colorbar.outline.set_linewidth(1.2)

    ax_box = fig.add_axes([0.215, 0.075, 0.515, 0.225])
    cluster_order = list(CLUSTER_COLORS)
    grouped = [
        cell_scores.loc[
            cell_scores["epithelial_cluster"].astype(str).eq(cluster), "cnv_score"
        ].to_numpy()
        for cluster in cluster_order
    ]
    boxplot = ax_box.boxplot(
        grouped,
        positions=np.arange(len(cluster_order)),
        widths=0.42,
        patch_artist=True,
        showfliers=False,
        boxprops={"linewidth": 0.75, "edgecolor": "#222222"},
        medianprops={"linewidth": 1.15, "color": "#111111"},
        whiskerprops={"linewidth": 0.75, "color": "#222222"},
        capprops={"linewidth": 0.75, "color": "#222222"},
    )
    for patch, cluster in zip(boxplot["boxes"], cluster_order):
        patch.set_facecolor(CLUSTER_COLORS[cluster])
        patch.set_alpha(0.30)

    rng = np.random.default_rng(SEED)
    for position, (cluster, values) in enumerate(zip(cluster_order, grouped)):
        jitter = rng.uniform(-0.16, 0.16, size=len(values))
        ax_box.scatter(
            np.full(len(values), position) + jitter,
            values,
            s=2.3,
            color=CLUSTER_COLORS[cluster],
            alpha=0.30,
            linewidths=0,
            rasterized=True,
            zorder=2,
        )
    y_upper = max(0.06, float(cell_scores["cnv_score"].quantile(0.995)) * 1.08)
    ax_box.set_ylim(0, y_upper)
    ax_box.set_xlim(-0.6, len(cluster_order) - 0.4)
    ax_box.set_xticks(np.arange(len(cluster_order)))
    ax_box.set_xticklabels(cluster_order, rotation=90, fontsize=7.4)
    ax_box.set_ylabel("Expression", fontsize=10)
    ax_box.set_title("cnv_score", fontsize=10.5, pad=5)
    ax_box.tick_params(axis="y", labelsize=7.2, width=0.8, length=3)
    ax_box.tick_params(axis="x", width=0.8, length=3)
    ax_box.spines["top"].set_visible(False)
    ax_box.spines["right"].set_visible(False)
    ax_box.spines["left"].set_linewidth(0.8)
    ax_box.spines["bottom"].set_linewidth(0.8)
    p_label = "p < 0.001" if overall_p < 0.001 else f"p = {overall_p:.3f}"
    ax_box.text(
        0.0,
        1.16,
        p_label,
        transform=ax_box.transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        fontstyle="italic",
    )

    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def main() -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    print("Parsing GENCODE v50 autosomal gene order...", flush=True)
    annotation = parse_gtf_gene_order()
    backed, fig3a, epi_index, ref_index = load_indices()
    print(
        f"Loading {len(epi_index):,} Figure 3A epithelial cells and "
        f"{len(ref_index):,} endothelial reference cells...",
        flush=True,
    )
    genes, gene_indices, reference_mean = ordered_expression_genes(
        backed, annotation, epi_index, ref_index
    )
    slices = chromosome_slices(genes)
    print(f"  retained {len(genes):,} expressed genes across {len(slices)} autosomes", flush=True)

    rng = np.random.default_rng(SEED)
    selected_epi = stratified_sample(fig3a, epi_index)
    selected_ref = rng.choice(ref_index, size=min(PLOT_REF_CELLS, len(ref_index)), replace=False)

    print("Estimating smoothed endothelial reference noise...", flush=True)
    _, raw_ref_retained, _, reference_sd = process_rows(
        backed.X,
        ref_index,
        gene_indices,
        reference_mean,
        slices,
        noise_floor=None,
        retain=set(selected_ref.tolist()),
    )
    noise_floor = reference_sd * NOISE_MULTIPLIER

    print("Computing epithelial expression-CNV profiles...", flush=True)
    epithelial_scores, epi_retained, _, _ = process_rows(
        backed.X,
        epi_index,
        gene_indices,
        reference_mean,
        slices,
        noise_floor=noise_floor,
        retain=set(selected_epi.tolist()),
    )
    reference_scores, ref_retained, _, _ = process_rows(
        backed.X,
        selected_ref,
        gene_indices,
        reference_mean,
        slices,
        noise_floor=noise_floor,
        retain=set(selected_ref.tolist()),
    )
    backed.file.close()

    epi_matrix, epi_order = order_heatmap_rows(selected_epi, epi_retained, fig3a, epi_index)
    ref_matrix = np.vstack([ref_retained[row] for row in selected_ref])
    ref_order_score = np.argsort(ref_matrix.mean(axis=1), kind="mergesort")
    ref_matrix = ref_matrix[ref_order_score]

    score_map = epithelial_scores.set_index("source_row")["cnv_score"]
    cell_scores = fig3a[
        ["cell_id", "epithelial_cluster", "disease_status", "patient_id", "sample_id"]
    ].copy()
    cell_scores["source_row"] = epi_index
    cell_scores["cnv_score"] = cell_scores["source_row"].map(score_map)
    groups = [
        cell_scores.loc[
            cell_scores["epithelial_cluster"].astype(str).eq(cluster), "cnv_score"
        ].to_numpy()
        for cluster in CLUSTER_COLORS
    ]
    overall_test = kruskal(*groups)
    cell_scores.drop(columns="source_row").to_csv(
        RESULT_DIR / "Fig3B_epithelial_cnv_scores.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    epi_order.to_csv(
        RESULT_DIR / "Fig3B_epithelial_heatmap_cell_order.csv",
        index=False,
        encoding="utf-8-sig",
    )
    genes.drop(columns="var_index").to_csv(
        RESULT_DIR / "Fig3B_epithelial_infercnv_gene_order.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    chromosome_table = pd.DataFrame(slices, columns=["chromosome", "start_index", "end_index"])
    chromosome_table["n_genes"] = chromosome_table["end_index"] - chromosome_table["start_index"]
    chromosome_table.to_csv(
        RESULT_DIR / "Fig3B_epithelial_chromosome_boundaries.csv",
        index=False,
        encoding="utf-8-sig",
    )
    np.savez_compressed(
        RESULT_DIR / "Fig3B_epithelial_heatmap_display_matrix.npz",
        epithelial=epi_matrix.astype(np.float32),
        reference=ref_matrix.astype(np.float32),
        genes=genes["gene"].to_numpy(dtype=str),
        chromosomes=genes["chromosome"].to_numpy(dtype=str),
    )

    metadata = {
        "source_object": str(H5AD),
        "figure3a_cell_table": str(FIG3A_CELLS),
        "reference_annotation": str(GTF),
        "reference_annotation_sha256": sha256(GTF),
        "reference_annotation_release": "GENCODE v50, GRCh38.p14, Ensembl 116",
        "dataset": DATASET,
        "observation_group": "Epithelial cells from Figure 3A",
        "reference_group": "GSE181919 Endothelial cells",
        "n_epithelial_cells": len(epi_index),
        "n_reference_cells": len(ref_index),
        "n_ordered_expressed_autosomal_genes": len(genes),
        "chromosomes": AUTOSOMES,
        "minimum_mean_log_expression": MIN_MEAN_EXPRESSION,
        "smoothing_window_genes": WINDOW_SIZE,
        "reference_noise_multiplier": NOISE_MULTIPLIER,
        "display_epithelial_cells": int(epi_matrix.shape[0]),
        "display_reference_cells": int(ref_matrix.shape[0]),
        "display_color_limits": [VMIN, VMAX],
        "cnv_score_overall_test": "Kruskal-Wallis across C0-C8",
        "cnv_score_kruskal_h": float(overall_test.statistic),
        "cnv_score_kruskal_p": float(overall_test.pvalue),
        "random_seed": SEED,
        "method_label": "expression-derived inferred CNV (inferCNV-style)",
        "interpretation_boundary": (
            "The heatmap is an expression-derived large-scale CNV proxy. It does "
            "not establish DNA-level copy-number alterations or malignancy without "
            "orthogonal genomic validation."
        ),
    }
    (RESULT_DIR / "Fig3B_epithelial_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_figure(
        epi_matrix,
        ref_matrix,
        genes,
        slices,
        cell_scores,
        float(overall_test.pvalue),
    )
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}", flush=True)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}", flush=True)
    print(
        f"CNV score median={cell_scores['cnv_score'].median():.4f}; "
        f"reference display median={reference_scores['cnv_score'].median():.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
