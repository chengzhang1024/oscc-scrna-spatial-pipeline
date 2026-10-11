"""Build epithelial Figure 3H: mean signaling entropy rate by cluster.

The primary panel reproduces the compact reference layout and uses the exact
single-cell SR values calculated for Figure 3G. Patient-balanced and technical
sensitivity summaries are exported for audit but are not added to the panel,
because the reference Figure 3H contains only cluster means.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


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

ROOT = Path(__file__).resolve().parent
FIG3G_CELLS = (
    ROOT / "results/paper9/figure3_epithelial/Fig3G_epithelial_cells_SR_diffusion.csv.gz"
)
FIG3G_CLUSTER_SUMMARY = (
    ROOT / "results/paper9/figure3_epithelial/Fig3G_epithelial_SR_by_cluster.csv"
)
RESULT_DIR = ROOT / "results/paper9/figure3_epithelial"
FIGURE_DIR = ROOT / "figures/paper9/figure3_epithelial"
OUTPUT_STEM = FIGURE_DIR / "Fig3H_epithelial_mean_SR_reference_style"

CLUSTERS = [f"C{i}" for i in range(9)]
QC_CLUSTERS = {"C5", "C8"}
SEED = 20260902
N_BOOTSTRAP = 10000
LINE_COLOR = "#3B6FA5"
MAX_COLOR = "#E63323"
AREA_COLOR = "#DCEAF2"
GRID_COLOR = "#D9D9D9"


def load_cells() -> pd.DataFrame:
    cells = pd.read_csv(FIG3G_CELLS)
    required = {
        "cell_id",
        "patient_id",
        "sample_id",
        "epithelial_cluster",
        "signaling_entropy_rate",
        "n_genes_by_counts",
        "total_counts",
    }
    if not required.issubset(cells.columns):
        raise RuntimeError(f"Figure 3G cells lack {sorted(required - set(cells.columns))}")
    if len(cells) != 5609 or cells["cell_id"].nunique() != 5609:
        raise RuntimeError("Figure 3G does not contain the fixed 5,609 unique cells")
    if set(cells["epithelial_cluster"].astype(str)) != set(CLUSTERS):
        raise RuntimeError("Figure 3G does not contain exactly C0-C8")
    for column in ("signaling_entropy_rate", "n_genes_by_counts", "total_counts"):
        cells[column] = pd.to_numeric(cells[column], errors="coerce")
    if cells[list(required - {"cell_id", "patient_id", "sample_id", "epithelial_cluster"})].isna().any().any():
        raise RuntimeError("Figure 3G contains missing SR or QC values")
    return cells


def summarize_cell_level(cells: pd.DataFrame) -> pd.DataFrame:
    grouped = cells.groupby("epithelial_cluster", observed=False)["signaling_entropy_rate"]
    summary = grouped.agg(["count", "mean", "median", "std", "min", "max"]).reindex(CLUSTERS)
    summary = summary.rename(
        columns={
            "count": "n_cells",
            "mean": "mean_SR",
            "median": "median_SR",
            "std": "sd_SR",
            "min": "min_SR",
            "max": "max_SR",
        }
    )
    summary["sem_SR"] = summary["sd_SR"] / np.sqrt(summary["n_cells"])
    summary["cell_level_ci95_low"] = summary["mean_SR"] - 1.96 * summary["sem_SR"]
    summary["cell_level_ci95_high"] = summary["mean_SR"] + 1.96 * summary["sem_SR"]
    summary["cell_level_mean_rank_desc"] = summary["mean_SR"].rank(
        ascending=False, method="min"
    ).astype(int)
    summary["is_highest_mean_SR"] = summary["mean_SR"].eq(summary["mean_SR"].max())
    summary["qc_identity_flag"] = [
        "mixed_mesenchymal_muscle_like"
        if cluster == "C5"
        else "immune_like"
        if cluster == "C8"
        else "none"
        for cluster in summary.index
    ]
    return summary.rename_axis("epithelial_cluster").reset_index()


def summarize_patient_balanced(cells: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    patient_cluster = (
        cells.groupby(["patient_id", "epithelial_cluster"], observed=False)[
            "signaling_entropy_rate"
        ]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(
            columns={
                "count": "n_cells",
                "mean": "patient_mean_SR",
                "median": "patient_median_SR",
            }
        )
    )
    patient_cluster.to_csv(
        RESULT_DIR / "Fig3H_epithelial_patient_cluster_SR.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rng = np.random.default_rng(SEED)
    rows: list[dict[str, float | int | str]] = []
    for cluster in CLUSTERS:
        group = patient_cluster[patient_cluster["epithelial_cluster"].eq(cluster)].copy()
        values = group["patient_mean_SR"].to_numpy(dtype=float)
        if len(values) == 0:
            raise RuntimeError(f"No patient summaries are available for {cluster}")
        bootstrap_indices = rng.integers(0, len(values), size=(N_BOOTSTRAP, len(values)))
        bootstrap_means = values[bootstrap_indices].mean(axis=1)
        rows.append(
            {
                "epithelial_cluster": cluster,
                "n_patients": int(group["patient_id"].nunique()),
                "n_cells": int(group["n_cells"].sum()),
                "min_cells_per_patient": int(group["n_cells"].min()),
                "max_cells_per_patient": int(group["n_cells"].max()),
                "equal_patient_weight_mean_SR": float(values.mean()),
                "equal_patient_weight_median_SR": float(np.median(values)),
                "patient_bootstrap_ci95_low": float(np.quantile(bootstrap_means, 0.025)),
                "patient_bootstrap_ci95_high": float(np.quantile(bootstrap_means, 0.975)),
            }
        )
    summary = pd.DataFrame(rows)
    summary["patient_balanced_mean_rank_desc"] = summary[
        "equal_patient_weight_mean_SR"
    ].rank(ascending=False, method="min").astype(int)
    return summary, patient_cluster


def complexity_adjusted_summary(cells: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    sr = cells["signaling_entropy_rate"].to_numpy(dtype=float)
    qc = np.column_stack(
        [
            np.log1p(cells["n_genes_by_counts"].to_numpy(dtype=float)),
            np.log1p(cells["total_counts"].to_numpy(dtype=float)),
        ]
    )
    qc = (qc - qc.mean(axis=0)) / np.maximum(qc.std(axis=0), 1e-12)
    design = np.column_stack([np.ones(len(cells)), qc])
    coefficients, *_ = np.linalg.lstsq(design, sr, rcond=None)
    residual = sr - design @ coefficients
    adjusted = cells[["epithelial_cluster"]].copy()
    adjusted["complexity_adjusted_SR_residual"] = residual
    summary = (
        adjusted.groupby("epithelial_cluster", observed=False)[
            "complexity_adjusted_SR_residual"
        ]
        .agg(["count", "mean", "median", "std"])
        .reindex(CLUSTERS)
        .reset_index()
        .rename(
            columns={
                "count": "n_cells",
                "mean": "mean_adjusted_SR_residual",
                "median": "median_adjusted_SR_residual",
                "std": "sd_adjusted_SR_residual",
            }
        )
    )
    summary["adjusted_mean_rank_desc"] = summary["mean_adjusted_SR_residual"].rank(
        ascending=False, method="min"
    ).astype(int)
    model = {
        "intercept": float(coefficients[0]),
        "z_log1p_n_genes_coefficient": float(coefficients[1]),
        "z_log1p_total_counts_coefficient": float(coefficients[2]),
        "r_squared": float(1.0 - np.sum(residual**2) / np.sum((sr - sr.mean()) ** 2)),
    }
    return summary, model


def combine_audit(
    cell_summary: pd.DataFrame,
    patient_summary: pd.DataFrame,
    adjusted_summary: pd.DataFrame,
) -> pd.DataFrame:
    combined = (
        cell_summary.merge(patient_summary, on=["epithelial_cluster", "n_cells"])
        .merge(adjusted_summary, on=["epithelial_cluster", "n_cells"])
        .sort_values("epithelial_cluster")
        .reset_index(drop=True)
    )
    combined["rank_shift_patient_minus_cell"] = (
        combined["patient_balanced_mean_rank_desc"]
        - combined["cell_level_mean_rank_desc"]
    )
    combined["rank_shift_adjusted_minus_cell"] = (
        combined["adjusted_mean_rank_desc"] - combined["cell_level_mean_rank_desc"]
    )
    return combined


def reference_axis_limits(values: np.ndarray) -> tuple[float, float, np.ndarray]:
    lower = np.floor((values.min() - 0.006) / 0.02) * 0.02
    upper = np.ceil((values.max() + 0.006) / 0.02) * 0.02
    lower = max(0.0, float(lower))
    upper = min(1.0, float(upper))
    ticks = np.arange(lower, upper + 1e-9, 0.02)
    return lower, upper, ticks


def plot_figure(summary: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    x = np.arange(len(CLUSTERS), dtype=float)
    y = summary.set_index("epithelial_cluster").loc[CLUSTERS, "mean_SR"].to_numpy()
    maximum_index = int(np.argmax(y))
    ymin, ymax, yticks = reference_axis_limits(y)

    fig = plt.figure(figsize=(5.70, 3.85), facecolor="white")
    ax = fig.add_axes([0.165, 0.190, 0.785, 0.655])
    ax.fill_between(x, ymin, y, color=AREA_COLOR, alpha=0.48, zorder=1)
    ax.plot(
        x,
        y,
        color=LINE_COLOR,
        linewidth=4.6,
        marker="o",
        markersize=8.0,
        markerfacecolor=LINE_COLOR,
        markeredgecolor=LINE_COLOR,
        solid_capstyle="round",
        solid_joinstyle="round",
        zorder=3,
    )
    ax.scatter(
        [x[maximum_index]],
        [y[maximum_index]],
        s=185,
        c=MAX_COLOR,
        edgecolors=MAX_COLOR,
        linewidths=0.8,
        zorder=5,
    )

    ax.set_xlim(-0.20, len(CLUSTERS) - 0.80)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(x)
    ax.set_xticklabels(CLUSTERS, fontsize=9.6)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{value:.2f}" for value in yticks], fontsize=8.8)
    ax.set_xlabel("Cluster", fontsize=12.0, fontweight="bold", labelpad=4)
    ax.set_ylabel("SR Value", fontsize=12.0, fontweight="bold", labelpad=6)
    ax.set_title(
        "Stemness Score (SR) Across Clusters",
        fontsize=14.0,
        fontweight="bold",
        pad=12,
    )
    ax.yaxis.grid(True, color=GRID_COLOR, linewidth=1.15)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", length=0, pad=5)
    ax.tick_params(axis="y", length=0, pad=4)
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.text(0.018, 0.970, "H", fontsize=27, fontweight="normal", va="top", ha="left")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def export_results(
    combined: pd.DataFrame,
    complexity_model: dict[str, float],
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    combined.to_csv(
        RESULT_DIR / "Fig3H_epithelial_mean_SR_and_sensitivity.csv",
        index=False,
        encoding="utf-8-sig",
    )

    prior = pd.read_csv(FIG3G_CLUSTER_SUMMARY).set_index("epithelial_cluster")
    current = combined.set_index("epithelial_cluster")
    maximum_abs_difference = float(
        np.max(np.abs(prior.loc[CLUSTERS, "mean_SR"] - current.loc[CLUSTERS, "mean_SR"]))
    )
    rho_patient = spearmanr(
        current.loc[CLUSTERS, "mean_SR"],
        current.loc[CLUSTERS, "equal_patient_weight_mean_SR"],
    )
    rho_adjusted = spearmanr(
        current.loc[CLUSTERS, "mean_SR"],
        current.loc[CLUSTERS, "mean_adjusted_SR_residual"],
    )
    top_cell = str(current["mean_SR"].idxmax())
    top_patient = str(current["equal_patient_weight_mean_SR"].idxmax())
    top_adjusted = str(current["mean_adjusted_SR_residual"].idxmax())
    clean = current.loc[[cluster for cluster in CLUSTERS if cluster not in QC_CLUSTERS]]

    metadata = {
        "figure3g_cell_table": str(FIG3G_CELLS),
        "figure3g_cluster_summary": str(FIG3G_CLUSTER_SUMMARY),
        "n_cells": int(combined["n_cells"].sum()),
        "clusters": CLUSTERS,
        "primary_statistic": "arithmetic mean single-cell LandSCENT SR within each C0-C8 cluster",
        "primary_highest_cluster": top_cell,
        "primary_highest_mean_SR": float(current.loc[top_cell, "mean_SR"]),
        "primary_lowest_cluster": str(current["mean_SR"].idxmin()),
        "primary_lowest_mean_SR": float(current["mean_SR"].min()),
        "exact_reuse_check_max_abs_difference_vs_Fig3G": maximum_abs_difference,
        "patient_balanced_sensitivity": {
            "definition": "equal weight to each patient-specific cluster mean",
            "bootstrap_resamples": N_BOOTSTRAP,
            "highest_cluster": top_patient,
            "spearman_rho_vs_primary_cluster_means": float(rho_patient.statistic),
            "spearman_p_value": float(rho_patient.pvalue),
        },
        "complexity_adjusted_sensitivity": {
            "definition": "cluster mean of residuals from SR ~ z(log1p detected genes) + z(log1p total counts)",
            "model": complexity_model,
            "highest_cluster": top_adjusted,
            "spearman_rho_vs_primary_cluster_means": float(rho_adjusted.statistic),
            "spearman_p_value": float(rho_adjusted.pvalue),
        },
        "qc_cluster_sensitivity": {
            "excluded_clusters": sorted(QC_CLUSTERS),
            "highest_remaining_cluster": str(clean["mean_SR"].idxmax()),
        },
        "display": {
            "cluster_order": CLUSTERS,
            "highest_point_color": MAX_COLOR,
            "line_color": LINE_COLOR,
            "uncertainty_not_drawn": "reference Figure 3H does not show error bars",
        },
        "visual_reference": "s41746-025-02297-4 Figure 3H",
        "user_decision": "C5 and C8 retained without exclusion",
        "random_seed": SEED,
        "interpretation_boundary": (
            "The line displays cell-level descriptive means and must not be treated as a patient-level "
            "inferential test. Cluster patient coverage is uneven, and Figure 3G documented strong SR "
            "correlation with sequencing complexity. C5 is mixed and C8 is immune-like."
        ),
    }
    (RESULT_DIR / "Fig3H_epithelial_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    print("Loading exact Figure 3G single-cell SR values...", flush=True)
    cells = load_cells()
    cell_summary = summarize_cell_level(cells)
    patient_summary, _ = summarize_patient_balanced(cells)
    adjusted_summary, complexity_model = complexity_adjusted_summary(cells)
    combined = combine_audit(cell_summary, patient_summary, adjusted_summary)
    print(
        combined[
            [
                "epithelial_cluster",
                "n_cells",
                "n_patients",
                "mean_SR",
                "equal_patient_weight_mean_SR",
                "mean_adjusted_SR_residual",
            ]
        ].to_string(index=False),
        flush=True,
    )
    export_results(combined, complexity_model)
    plot_figure(combined)
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}", flush=True)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
