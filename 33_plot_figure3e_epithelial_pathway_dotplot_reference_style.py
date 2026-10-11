"""Build Figure 3E: reference-style epithelial pathway marker dot plot.

Rows are the fixed Figure 3A clusters C0-C8. Dot area reports the fraction of
cells expressing a gene, and color reports the per-gene scaled cluster mean.
C5 and C8 are intentionally retained following the user's explicit decision.
"""

from __future__ import annotations

import json
from collections import OrderedDict
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
import numpy as np
import pandas as pd
from scipy import sparse


matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 0.9,
    }
)

ROOT = Path(__file__).resolve().parent
H5AD = ROOT / "results/integrated_3ds_annotated_paper9.h5ad"
FIG3A_CELLS = ROOT / "results/paper9/figure3_epithelial/Fig3A_epithelial_cells_and_umap.csv.gz"
RESULT_DIR = ROOT / "results/paper9/figure3_epithelial"
FIGURE_DIR = ROOT / "figures/paper9/figure3_epithelial"
OUTPUT_STEM = FIGURE_DIR / "Fig3E_epithelial_pathway_dotplot_reference_style"
CLUSTERS = [f"C{i}" for i in range(9)]

PROGRAMS = OrderedDict(
    [
        ("Keratinization", ["KRT5", "KRT14", "KRT15", "KRT19", "KRT13", "IVL", "SFN"]),
        ("EMT", ["VIM", "LAMC2", "ITGA6", "COL17A1", "MMP7", "SNAI2", "ZEB1"]),
        (
            "IFN / inflammation",
            ["STAT1", "ISG15", "IFITM1", "IFIT1", "CXCL10", "HLA-DRA", "PSMB9"],
        ),
        (
            "Hypoxia / glycolysis",
            ["HIF1A", "CA9", "SLC2A1", "ENO1", "LDHA", "PDK1", "PGK1"],
        ),
        ("DNA repair", ["PCNA", "PARP1", "RAD51", "BRCA1", "BRCA2", "MSH2", "MSH6"]),
        (
            "Cell cycle",
            ["CCNA2", "CCNB1", "CCND1", "CDK1", "CDK4", "MKI67", "TOP2A"],
        ),
    ]
)

GENE_AUDIT_NOTES = {
    "PARP1": "Used in place of requested FEN1 because FEN1 is absent from the integrated object",
}

COLOR_MAP = LinearSegmentedColormap.from_list(
    "reference_mean_expression",
    ["#FFF7F1", "#FCAE91", "#FB6A4A", "#DE2D26", "#7F0000"],
)
FRACTION_LEGEND = [0.10, 0.20, 0.30, 0.40, 0.50]


def load_expression() -> tuple[pd.DataFrame, sparse.csr_matrix, np.ndarray]:
    cells = pd.read_csv(FIG3A_CELLS)
    if set(cells["epithelial_cluster"].astype(str)) != set(CLUSTERS):
        raise RuntimeError("Figure 3A does not contain exactly C0-C8")
    backed = ad.read_h5ad(H5AD, backed="r")
    lookup = pd.Series(np.arange(backed.n_obs), index=backed.obs_names.astype(str))
    rows = lookup.reindex(cells["cell_id"].astype(str)).to_numpy()
    if np.isnan(rows).any():
        backed.file.close()
        raise RuntimeError("Some Figure 3A cells are absent from the source object")
    all_genes = [gene for genes in PROGRAMS.values() for gene in genes]
    var_lookup = pd.Series(np.arange(backed.n_vars), index=backed.var_names.astype(str))
    missing = [gene for gene in all_genes if gene not in var_lookup.index]
    if missing:
        backed.file.close()
        raise RuntimeError(f"Program genes absent from source object: {missing}")
    columns = var_lookup.reindex(all_genes).to_numpy(dtype=int)
    matrix = backed.X[rows.astype(int), :]
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    matrix = sparse.csr_matrix(matrix[:, columns], dtype=np.float32)
    backed.file.close()
    return cells, matrix, np.asarray(all_genes, dtype=str)


def calculate_dot_table(
    cells: pd.DataFrame,
    matrix: sparse.csr_matrix,
    genes: np.ndarray,
) -> pd.DataFrame:
    labels = cells["epithelial_cluster"].astype(str).to_numpy()
    means = np.empty((len(CLUSTERS), len(genes)), dtype=np.float32)
    fractions = np.empty_like(means)
    for cluster_index, cluster in enumerate(CLUSTERS):
        block = matrix[labels == cluster]
        means[cluster_index] = np.asarray(block.mean(axis=0)).ravel()
        fractions[cluster_index] = np.asarray(block.getnnz(axis=0)).ravel() / block.shape[0]

    minimum = means.min(axis=0, keepdims=True)
    maximum = means.max(axis=0, keepdims=True)
    scaled = (means - minimum) / (maximum - minimum + 1e-6)
    program_lookup = {
        gene: program for program, program_genes in PROGRAMS.items() for gene in program_genes
    }
    rows: list[dict[str, float | int | str]] = []
    for cluster_index, cluster in enumerate(CLUSTERS):
        cluster_n_cells = int((labels == cluster).sum())
        for gene_index, gene in enumerate(genes):
            rows.append(
                {
                    "epithelial_cluster": cluster,
                    "cluster_index": cluster_index,
                    "n_cells_in_cluster": cluster_n_cells,
                    "program": program_lookup[gene],
                    "gene": gene,
                    "gene_index": gene_index,
                    "mean_log_expression": float(means[cluster_index, gene_index]),
                    "scaled_mean_expression": float(scaled[cluster_index, gene_index]),
                    "fraction_expressing": float(fractions[cluster_index, gene_index]),
                }
            )
    return pd.DataFrame(rows)


def marker_size(fraction: np.ndarray | float) -> np.ndarray:
    values = np.asarray(fraction, dtype=float)
    # Reference legend is calibrated from 10% to 50%; values above 50% retain
    # the maximum size while their exact fractions remain in the audit table.
    clipped = np.clip(values, 0.0, 0.50)
    return 4.0 + (clipped / 0.50) * 105.0


def draw_program_bracket(
    ax: plt.Axes,
    start: int,
    end: int,
    label: str,
) -> None:
    y_bottom = -1.22
    y_top = -1.54
    ax.plot(
        [start - 0.42, start - 0.42, end + 0.42, end + 0.42],
        [y_bottom, y_top, y_top, y_bottom],
        color="black",
        lw=1.1,
        clip_on=False,
        solid_joinstyle="miter",
    )
    ax.text(
        (start + end) / 2,
        y_top - 0.15,
        label,
        ha="center",
        va="bottom",
        fontsize=8.7,
        clip_on=False,
    )


def plot_figure(dot_table: pd.DataFrame, genes: np.ndarray) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(13.8, 4.15), facecolor="white")
    ax = fig.add_axes([0.065, 0.245, 0.755, 0.565])

    x = dot_table["gene_index"].to_numpy(dtype=float)
    y = dot_table["cluster_index"].to_numpy(dtype=float)
    ax.scatter(
        x,
        y,
        s=marker_size(dot_table["fraction_expressing"].to_numpy()),
        c=dot_table["scaled_mean_expression"].to_numpy(),
        cmap=COLOR_MAP,
        vmin=0,
        vmax=1,
        edgecolors="#4A4A4A",
        linewidths=0.45,
        alpha=0.98,
    )
    ax.set_xlim(-0.65, len(genes) - 0.35)
    ax.set_ylim(len(CLUSTERS) - 0.45, -0.55)
    ax.set_xticks(np.arange(len(genes)))
    ax.set_xticklabels(genes, rotation=90, fontsize=7.1, ha="center", va="top")
    ax.set_yticks(np.arange(len(CLUSTERS)))
    ax.set_yticklabels(CLUSTERS, fontsize=8.0)
    ax.tick_params(axis="x", length=3.0, width=0.8, pad=3)
    ax.tick_params(axis="y", length=0, width=0.8, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.9)
        spine.set_color("black")

    offset = 0
    for program, program_genes in PROGRAMS.items():
        draw_program_bracket(ax, offset, offset + len(program_genes) - 1, program)
        offset += len(program_genes)

    size_axis = fig.add_axes([0.850, 0.405, 0.070, 0.365])
    size_axis.set_xlim(-0.65, 0.65)
    size_axis.set_ylim(-0.65, 4.65)
    size_axis.axis("off")
    size_axis.text(
        0.0,
        4.30,
        "Fraction of cells\nin group (%)",
        fontsize=7.0,
        ha="center",
        va="top",
    )
    for index, fraction in enumerate(FRACTION_LEGEND):
        size_axis.scatter(
            [0.0],
            [3.45 - index * 0.70],
            s=marker_size(fraction),
            facecolors="#707070",
            edgecolors="#4A4A4A",
            linewidths=0.4,
            zorder=5,
        )
        size_axis.text(
            0.48,
            3.45 - index * 0.70,
            str(int(fraction * 100)),
            fontsize=6.4,
            ha="left",
            va="center",
        )

    color_axis = fig.add_axes([0.940, 0.405, 0.018, 0.300])
    gradient = np.linspace(0, 1, 256)[:, None]
    color_axis.imshow(gradient, aspect="auto", cmap=COLOR_MAP, vmin=0, vmax=1)
    color_axis.set_xticks([])
    color_axis.set_yticks([0, 128, 255])
    color_axis.set_yticklabels(["0.0", "0.5", "1.0"], fontsize=6.2)
    color_axis.yaxis.tick_right()
    color_axis.tick_params(axis="y", length=2.2, width=0.5, pad=2.0)
    for spine in color_axis.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.65)
    fig.text(
        0.949,
        0.735,
        "Mean expression\nin group",
        fontsize=7.0,
        ha="center",
        va="bottom",
    )

    fig.text(0.012, 0.965, "E", fontsize=24, fontweight="normal", va="top", ha="left")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def export_results(dot_table: pd.DataFrame, genes: np.ndarray) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    dot_table.to_csv(
        RESULT_DIR / "Fig3E_epithelial_pathway_dotplot_values.csv",
        index=False,
        encoding="utf-8-sig",
    )
    audit_rows = []
    gene_index = 0
    for program, program_genes in PROGRAMS.items():
        for gene in program_genes:
            audit_rows.append(
                {
                    "program": program,
                    "gene": gene,
                    "gene_index": gene_index,
                    "present_in_object": True,
                    "note": GENE_AUDIT_NOTES.get(gene, ""),
                }
            )
            gene_index += 1
    audit_rows.append(
        {
            "program": "DNA repair",
            "gene": "FEN1",
            "gene_index": np.nan,
            "present_in_object": False,
            "note": "Absent from integrated object; replaced by PARP1",
        }
    )
    pd.DataFrame(audit_rows).to_csv(
        RESULT_DIR / "Fig3E_epithelial_pathway_gene_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "source_object": str(H5AD),
        "figure3a_cell_table": str(FIG3A_CELLS),
        "n_cells": int(
            dot_table.drop_duplicates("epithelial_cluster")["n_cells_in_cluster"].sum()
        ),
        "clusters": CLUSTERS,
        "programs": PROGRAMS,
        "n_programs": len(PROGRAMS),
        "n_genes": len(genes),
        "dot_size": "fraction of cells with log-normalized expression > 0; display capped at 50%",
        "dot_color": "per-gene min-max scaled mean log-normalized expression across C0-C8",
        "color_range": [0.0, 1.0],
        "visual_reference": "s41746-025-02297-4 Figure 3E",
        "user_decision": "C5 and C8 retained without exclusion",
        "interpretation_boundary": (
            "C8 retains a strong immune-like program and C5 retains a mixed "
            "mesenchymal-muscle-like program. Their dot-plot signals are displayed "
            "transparently but must not be interpreted as validated epithelial states."
        ),
    }
    (RESULT_DIR / "Fig3E_epithelial_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    print("Loading Figure 3A epithelial cells and program genes...", flush=True)
    cells, matrix, genes = load_expression()
    print(f"  cells={len(cells):,}; programs={len(PROGRAMS)}; genes={len(genes)}", flush=True)
    dot_table = calculate_dot_table(cells, matrix, genes)
    export_results(dot_table, genes)
    plot_figure(dot_table, genes)
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}", flush=True)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
