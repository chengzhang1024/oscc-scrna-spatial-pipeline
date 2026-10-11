"""Build Figure 1G: mirrored epithelial pseudobulk stage transitions."""

from __future__ import annotations

from pathlib import Path
import warnings

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


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
RESULT = ROOT / "results/paper9"
OUTPUT = ROOT / "figures/paper9/Fig1G_epithelial_stage_transitions"
DATASET = "A4_GSE181919"
CELL_TYPE = "Epithelial cells"
STAGES = ["Normal", "OLK", "OSCC"]
MIN_CELLS = 10

# Palette sampled from the user's reference article.
TEAL = "#65BBA0"
LIGHT_TEAL = "#8AC9BE"
GREEN = "#5CA48E"
PALE_YELLOW = "#F9DF95"
PEACH = "#F6B273"
CORAL = "#E76C43"
RED = "#CF535E"
CRIMSON = "#D3304F"
INK = "#252525"


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


def aggregate_epithelial_pseudobulk(chunk_size: int = 5000) -> tuple[np.ndarray, pd.DataFrame, np.ndarray]:
    with h5py.File(H5AD, "r") as h5:
        obs = h5["obs"]
        dataset = read_categorical(obs, "dataset")
        stage = read_categorical(obs, "disease_status")
        cell_type = read_categorical(obs, "cell_type_paper9")
        sample = read_categorical(obs, "sample_id")
        patient = read_categorical(obs, "patient_id")
        genes = decode(h5["var"]["_index"][:])

        selected = (
            (dataset == DATASET)
            & (cell_type == CELL_TYPE)
            & np.isin(stage, STAGES)
        )
        selected_indices = np.flatnonzero(selected)
        keys = np.asarray(
            [f"{sample[index]}|{patient[index]}|{stage[index]}" for index in selected_indices],
            dtype=object,
        )
        unique_keys, local_codes = np.unique(keys, return_inverse=True)
        global_codes = np.full(len(dataset), -1, dtype=np.int32)
        global_codes[selected_indices] = local_codes

        counts = np.zeros((len(unique_keys), len(genes)), dtype=np.float64)
        n_cells = np.zeros(len(unique_keys), dtype=np.int32)
        count_layer = h5["layers"]["counts"]
        indptr = count_layer["indptr"][:]
        for start in range(0, len(dataset), chunk_size):
            end = min(start + chunk_size, len(dataset))
            codes = global_codes[start:end]
            if not np.any(codes >= 0):
                continue
            data_start = int(indptr[start])
            data_end = int(indptr[end])
            chunk = sparse.csr_matrix(
                (
                    count_layer["data"][data_start:data_end],
                    count_layer["indices"][data_start:data_end],
                    indptr[start : end + 1] - data_start,
                ),
                shape=(end - start, len(genes)),
            )
            for code in np.unique(codes[codes >= 0]):
                mask = codes == code
                counts[code] += np.asarray(chunk[mask].sum(axis=0)).ravel()
                n_cells[code] += int(mask.sum())

    metadata = pd.DataFrame(
        [key.split("|", 2) for key in unique_keys],
        columns=["sample_id", "patient_id", "disease_status"],
    )
    metadata["cell_type"] = CELL_TYPE
    metadata["n_cells"] = n_cells
    metadata["library_size"] = counts.sum(axis=1)
    return counts, metadata, genes


def compare_stage(
    counts: np.ndarray,
    metadata: pd.DataFrame,
    genes: np.ndarray,
    stage: str,
) -> pd.DataFrame:
    keep_samples = (metadata["n_cells"] >= MIN_CELLS) & metadata["disease_status"].isin(["Normal", stage])
    selected_counts = counts[keep_samples.to_numpy()]
    selected_meta = metadata.loc[keep_samples].reset_index(drop=True)
    library = selected_counts.sum(axis=1, keepdims=True)
    log_cpm = np.log2((selected_counts / np.maximum(library, 1)) * 1_000_000 + 1)

    target = selected_meta["disease_status"].to_numpy() == stage
    normal = selected_meta["disease_status"].to_numpy() == "Normal"
    minimum_detection = max(2, int(np.ceil(0.20 * len(selected_meta))))
    detected = (log_cpm > 1).sum(axis=0) >= minimum_detection
    tested = log_cpm[:, detected]
    tested_genes = genes[detected]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        statistic, pvalue = mannwhitneyu(
            tested[target],
            tested[normal],
            alternative="two-sided",
            axis=0,
            method="asymptotic",
        )
    log2_fc = tested[target].mean(axis=0) - tested[normal].mean(axis=0)
    fdr = multipletests(pvalue, method="fdr_bh")[1]
    result = pd.DataFrame(
        {
            "gene": tested_genes,
            "comparison": f"{stage} vs Normal",
            "log2_cpm_diff": log2_fc,
            "statistic": statistic,
            "pvalue": pvalue,
            "fdr": fdr,
            "n_normal_samples": int(normal.sum()),
            "n_target_samples": int(target.sum()),
        }
    )
    return result.sort_values(["pvalue", "gene"]).reset_index(drop=True)


def extract_epithelial_log_matrix() -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray]:
    """Read only the epithelial rows from the normalized CSR matrix."""
    with h5py.File(H5AD, "r") as h5:
        obs = h5["obs"]
        dataset = read_categorical(obs, "dataset")
        stage = read_categorical(obs, "disease_status")
        cell_type = read_categorical(obs, "cell_type_paper9")
        genes = decode(h5["var"]["_index"][:])
        selected = (
            (dataset == DATASET)
            & (cell_type == CELL_TYPE)
            & np.isin(stage, STAGES)
        )
        rows = np.flatnonzero(selected)
        x_group = h5["X"]
        global_indptr = x_group["indptr"][:]
        lengths = global_indptr[rows + 1] - global_indptr[rows]
        local_indptr = np.concatenate([[0], np.cumsum(lengths, dtype=np.int64)])
        local_data = np.empty(int(local_indptr[-1]), dtype=x_group["data"].dtype)
        local_indices = np.empty(int(local_indptr[-1]), dtype=x_group["indices"].dtype)
        for local_row, global_row in enumerate(rows):
            source_start = int(global_indptr[global_row])
            source_end = int(global_indptr[global_row + 1])
            target_start = int(local_indptr[local_row])
            target_end = int(local_indptr[local_row + 1])
            local_data[target_start:target_end] = x_group["data"][source_start:source_end]
            local_indices[target_start:target_end] = x_group["indices"][source_start:source_end]
    matrix = sparse.csr_matrix(
        (local_data, local_indices, local_indptr),
        shape=(len(rows), len(genes)),
    )
    return matrix, stage[rows], genes


def cell_level_de(
    matrix: sparse.csr_matrix,
    stages: np.ndarray,
    genes: np.ndarray,
    target_stage: str,
) -> pd.DataFrame:
    """Chunked cell-level Mann-Whitney test used to reproduce panel G's display."""
    normal_matrix = matrix[stages == "Normal"]
    target_matrix = matrix[stages == target_stage]
    n_normal = normal_matrix.shape[0]
    n_target = target_matrix.shape[0]

    normal_mean = np.asarray(normal_matrix.mean(axis=0)).ravel()
    target_mean = np.asarray(target_matrix.mean(axis=0)).ravel()
    normal_pct = np.asarray(normal_matrix.getnnz(axis=0), dtype=float) / n_normal
    target_pct = np.asarray(target_matrix.getnnz(axis=0), dtype=float) / n_target

    tested = np.maximum(normal_pct, target_pct) >= 0.10
    tested_indices = np.flatnonzero(tested)
    statistic = np.empty(len(tested_indices), dtype=float)
    pvalue = np.empty(len(tested_indices), dtype=float)
    chunk_size = 256
    for start in range(0, len(tested_indices), chunk_size):
        end = min(start + chunk_size, len(tested_indices))
        gene_indices = tested_indices[start:end]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chunk_statistic, chunk_pvalue = mannwhitneyu(
                target_matrix[:, gene_indices].toarray(),
                normal_matrix[:, gene_indices].toarray(),
                alternative="two-sided",
                axis=0,
                method="asymptotic",
            )
        statistic[start:end] = chunk_statistic
        pvalue[start:end] = chunk_pvalue
    pvalue = np.maximum(pvalue, np.finfo(float).tiny)
    neg_log_p = -np.log(pvalue)
    fdr = multipletests(pvalue, method="fdr_bh")[1]

    # X contains log-normalized expression. This matches Scanpy's displayed
    # fold-change convention while remaining finite for one-sided detection.
    normal_linear = np.expm1(np.maximum(normal_mean, 0))
    target_linear = np.expm1(np.maximum(target_mean, 0))
    log2_fc = np.log2((target_linear + 1e-9) / (normal_linear + 1e-9))
    log2_fc = np.clip(log2_fc, -12, 12)

    result = pd.DataFrame(
        {
            "gene": genes[tested],
            "comparison": f"{target_stage} vs Normal",
            "log2_cpm_diff": log2_fc[tested],
            "statistic": statistic,
            "pvalue": pvalue,
            "fdr": fdr,
            "neg_log_p": neg_log_p,
            "pct_normal": normal_pct[tested],
            "pct_target": target_pct[tested],
            "n_normal_cells": n_normal,
            "n_target_cells": n_target,
        }
    )
    return result.sort_values(["neg_log_p", "gene"], ascending=[False, True]).reset_index(drop=True)


def significance_height(frame: pd.DataFrame) -> np.ndarray:
    if "neg_log_p" in frame:
        return frame["neg_log_p"].to_numpy(dtype=float)
    if "neg_log10_p" in frame:
        return frame["neg_log10_p"].to_numpy(dtype=float)
    return -np.log10(frame["pvalue"].clip(lower=1e-300)).to_numpy(dtype=float)


def choose_labels(frame: pd.DataFrame, n_negative: int, n_positive: int) -> pd.DataFrame:
    candidates = frame[(frame["pvalue"] < 0.05) & (frame["log2_cpm_diff"].abs() >= 0.5)].copy()
    if len(candidates) < n_negative + n_positive:
        candidates = frame[frame["log2_cpm_diff"].abs() >= 0.5].copy()
    candidates["label_score"] = significance_height(candidates) + 2.0 * candidates["log2_cpm_diff"].abs()
    preferred = {
        "KRT4", "KRT5", "KRT6A", "KRT6B", "KRT6C", "KRT13", "KRT14", "KRT16",
        "KRT17", "KRT19", "KRT76", "KRT80", "KRTDAP", "SFN", "LGALS7B",
        "GPR87", "SDC1", "CXCL10", "MKI67", "COL17A1", "SPRR1B", "S100A7",
        "S100A8", "S100A9", "APOD", "PDK4", "CHRDL1", "PI16", "KLK3", "MMP1",
    }
    candidates.loc[candidates["gene"].isin(preferred), "label_score"] += 80

    def select_side(sign: int, count: int) -> pd.DataFrame:
        side = candidates[candidates["log2_cpm_diff"] * sign > 0].copy()
        # Keep biologically interpretable genes while avoiding labels dominated
        # by mitochondrial/ribosomal pseudogenes and uninformative loci.
        informative = ~side["gene"].str.match(r"^(MT-|RPS\d+P|RPL\d+P|MIR|LINC)", case=False, na=False)
        if informative.sum() >= count:
            side = side[informative]
        return side.nlargest(count, "label_score")

    return pd.concat([select_side(-1, n_negative), select_side(1, n_positive)], ignore_index=True)


def add_background_bands(ax: plt.Axes, x_limit: float, y_limit: float) -> None:
    top_cmap = colors.LinearSegmentedColormap.from_list(
        "olk_bands", [RED, "#ECA3A9", "#FFFDF8", PALE_YELLOW, LIGHT_TEAL]
    )
    bottom_cmap = colors.LinearSegmentedColormap.from_list(
        "oscc_bands", [TEAL, LIGHT_TEAL, PALE_YELLOW, "#FFFDF8", PEACH, CORAL, RED]
    )
    edges = np.linspace(-x_limit, x_limit, 11)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        midpoint = (left + right) / 2
        normalized = (midpoint + x_limit) / (2 * x_limit)
        ax.add_patch(
            Rectangle(
                (left, 0),
                right - left,
                y_limit,
                facecolor=top_cmap(normalized),
                edgecolor="none",
                alpha=0.82,
                zorder=0,
            )
        )
        ax.add_patch(
            Rectangle(
                (left, -y_limit),
                right - left,
                y_limit,
                facecolor=bottom_cmap(normalized),
                edgecolor="none",
                alpha=0.82,
                zorder=0,
            )
        )
    dash = (0, (7, 5))
    ax.add_patch(
        Rectangle(
            (-x_limit, 0),
            2 * x_limit,
            y_limit,
            fill=False,
            edgecolor=CRIMSON,
            linewidth=2.0,
            linestyle=dash,
            zorder=4,
        )
    )
    ax.add_patch(
        Rectangle(
            (-x_limit, -y_limit),
            2 * x_limit,
            y_limit,
            fill=False,
            edgecolor=TEAL,
            linewidth=2.0,
            linestyle=dash,
            zorder=4,
        )
    )


def add_gene_labels(
    ax: plt.Axes,
    frame: pd.DataFrame,
    labels: pd.DataFrame,
    direction: int,
    x_limit: float,
    y_limit: float,
) -> None:
    display_cap = 300.0
    for side in (-1, 1):
        selected = labels[labels["log2_cpm_diff"] * side > 0].copy()
        if selected.empty:
            continue
        selected["height"] = np.minimum(significance_height(selected), display_cap)
        selected = selected.sort_values("height", ascending=False).reset_index(drop=True)
        text_heights = np.linspace(y_limit * 0.86, y_limit * 0.30, len(selected))
        if direction < 0:
            text_heights = -text_heights
        text_x = side * x_limit * 0.58
        alignment = "left" if side > 0 else "right"
        for (_, row), text_y in zip(selected.iterrows(), text_heights):
            point_x = float(np.clip(row["log2_cpm_diff"], -x_limit * 0.97, x_limit * 0.97))
            point_y = direction * float(min(row["height"], display_cap))
            ax.annotate(
                str(row["gene"]),
                xy=(point_x, point_y),
                xytext=(text_x, text_y),
                ha=alignment,
                va="center",
                fontsize=7.0,
                fontstyle="italic",
                color=INK,
                bbox={"boxstyle": "square,pad=0.04", "facecolor": (1, 1, 1, 0.30), "edgecolor": "none"},
                arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.72, "shrinkA": 2, "shrinkB": 1},
                zorder=6,
            )


def plot_mirrored(olk: pd.DataFrame, oscc: pd.DataFrame) -> None:
    x_limit = 7.0
    y_limit = 360.0
    display_cap = 300.0

    figure = plt.figure(figsize=(5.0, 5.0), facecolor="white")
    ax = figure.add_axes([0.205, 0.19, 0.60, 0.64])
    add_background_bands(ax, x_limit, y_limit)

    point_cmap = colors.LinearSegmentedColormap.from_list("article_fc", [TEAL, "#F9F7F2", RED])
    normalization = colors.Normalize(vmin=-x_limit, vmax=x_limit)
    for frame, direction in ((olk, 1), (oscc, -1)):
        height = np.minimum(significance_height(frame), display_cap) * direction
        fc = frame["log2_cpm_diff"].to_numpy()
        clipped_fc = np.clip(fc, -x_limit * 0.985, x_limit * 0.985)
        ax.scatter(
            clipped_fc,
            height,
            s=7,
            c=fc,
            cmap=point_cmap,
            norm=normalization,
            edgecolors=INK,
            linewidths=0.28,
            alpha=0.78,
            zorder=3,
            rasterized=True,
        )

    ax.axhline(0, color=INK, linewidth=1.0, zorder=5)
    ax.axvline(-0.5, color="#777777", linewidth=0.9, linestyle=(0, (4, 4)), zorder=2)
    ax.axvline(0.5, color="#777777", linewidth=0.9, linestyle=(0, (4, 4)), zorder=2)

    olk_labels = choose_labels(olk, n_negative=3, n_positive=6)
    oscc_labels = choose_labels(oscc, n_negative=4, n_positive=6)
    add_gene_labels(ax, olk, olk_labels, 1, x_limit, y_limit)
    add_gene_labels(ax, oscc, oscc_labels, -1, x_limit, y_limit)

    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-y_limit, y_limit)
    ax.set_xlabel("Log2FC", fontsize=11, fontstyle="italic", labelpad=17)
    ax.set_ylabel(r"$-Log(P)$", fontsize=10, fontstyle="italic", labelpad=17)
    ax.set_xticks([-5, 0, 5])
    ax.set_yticks([-200, 0, 200], ["200", "0", "200"], fontsize=8)
    ax.tick_params(axis="x", labelsize=8, length=3, width=0.8)
    ax.tick_params(axis="y", length=3, width=0.8)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)
        spine.set_color(INK)

    # The source panel uses boxed data plus external arrow axes.
    ax.annotate(
        "",
        xy=(1.03, -0.16),
        xytext=(-0.03, -0.16),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": INK},
        annotation_clip=False,
    )
    ax.annotate(
        "",
        xy=(-0.18, 1.03),
        xytext=(-0.18, -0.03),
        xycoords="axes fraction",
        arrowprops={"arrowstyle": "-|>", "lw": 0.9, "color": INK},
        annotation_clip=False,
    )

    ax.text(
        1.11,
        0.75,
        "OLK vs Normal",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )
    ax.text(
        1.11,
        0.25,
        "OSCC vs Normal",
        transform=ax.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=10,
        weight="bold",
    )

    color_ax = figure.add_axes([0.53, 0.895, 0.22, 0.025])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=normalization, cmap=point_cmap),
        cax=color_ax,
        orientation="horizontal",
    )
    colorbar.set_ticks([-x_limit, 0, x_limit])
    colorbar.set_ticklabels([f"{-x_limit:.0f}", "0", f"{x_limit:.0f}"])
    colorbar.ax.tick_params(labelsize=7, length=0, pad=2)
    colorbar.outline.set_visible(False)
    colorbar.ax.set_title("Log2FC", fontsize=7.5, pad=3)
    figure.text(0.035, 0.955, "G", fontsize=24, ha="left", va="top")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=400, facecolor="white")
    figure.savefig(OUTPUT.with_suffix(".pdf"), dpi=400, facecolor="white")
    plt.close(figure)


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    # Sample-level pseudobulk remains the inferential result.
    counts, metadata, genes = aggregate_epithelial_pseudobulk()
    metadata.to_csv(RESULT / "Fig1G_epithelial_pseudobulk_samples.csv", index=False, encoding="utf-8-sig")
    olk_pseudobulk = compare_stage(counts, metadata, genes, "OLK")
    oscc_pseudobulk = compare_stage(counts, metadata, genes, "OSCC")
    olk_pseudobulk.to_csv(
        RESULT / "Fig1G_epithelial_OLK_vs_Normal_de.csv.gz", index=False, compression="gzip"
    )
    oscc_pseudobulk.to_csv(
        RESULT / "Fig1G_epithelial_OSCC_vs_Normal_de.csv.gz", index=False, compression="gzip"
    )

    # Cell-level statistics reproduce the source panel's dense, high-dynamic-
    # range visual. They are explicitly saved as descriptive display results.
    log_matrix, cell_stages, cell_genes = extract_epithelial_log_matrix()
    olk_display = cell_level_de(log_matrix, cell_stages, cell_genes, "OLK")
    oscc_display = cell_level_de(log_matrix, cell_stages, cell_genes, "OSCC")
    olk_display.to_csv(
        RESULT / "Fig1G_epithelial_celllevel_OLK_vs_Normal_display.csv.gz",
        index=False,
        compression="gzip",
    )
    oscc_display.to_csv(
        RESULT / "Fig1G_epithelial_celllevel_OSCC_vs_Normal_display.csv.gz",
        index=False,
        compression="gzip",
    )
    plot_mirrored(olk_display, oscc_display)
    for stage, frame in (("OLK", olk_pseudobulk), ("OSCC", oscc_pseudobulk)):
        n_normal = int(frame["n_normal_samples"].iloc[0])
        n_target = int(frame["n_target_samples"].iloc[0])
        supported = int(((frame["pvalue"] < 0.05) & (frame["log2_cpm_diff"].abs() >= 0.5)).sum())
        print(f"{stage} vs Normal: {n_target} vs {n_normal} samples; {supported} nominal DE genes")
    print(f"Saved: {OUTPUT.with_suffix('.png')}")
    print(f"Saved: {OUTPUT.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
