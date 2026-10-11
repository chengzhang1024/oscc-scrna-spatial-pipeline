"""Build epithelial Figure 3I: single-cell SR distributions across C0-C8."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal


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
FIG3G_CELLS = ROOT / "results/paper9/figure3_epithelial/Fig3G_epithelial_cells_SR_diffusion.csv.gz"
RESULT_DIR = ROOT / "results/paper9/figure3_epithelial"
FIGURE_DIR = ROOT / "figures/paper9/figure3_epithelial"
OUTPUT_STEM = FIGURE_DIR / "Fig3I_epithelial_SR_distribution_reference_style"
CLUSTERS = [f"C{i}" for i in range(9)]
PALETTE = {
    "C0": "#17365D",
    "C1": "#D9D9D9",
    "C2": "#294E75",
    "C3": "#3A719E",
    "C4": "#102D4D",
    "C5": "#284D70",
    "C6": "#5AA6C8",
    "C7": "#244568",
    "C8": "#123354",
}


def load_data() -> pd.DataFrame:
    cells = pd.read_csv(FIG3G_CELLS)
    if len(cells) != 5609 or set(cells["epithelial_cluster"].astype(str)) != set(CLUSTERS):
        raise RuntimeError("Figure 3G is not the fixed 5,609-cell C0-C8 input")
    cells["signaling_entropy_rate"] = pd.to_numeric(
        cells["signaling_entropy_rate"], errors="coerce"
    )
    if cells["signaling_entropy_rate"].isna().any():
        raise RuntimeError("Figure 3G SR values contain missing entries")
    return cells


def summarize(cells: pd.DataFrame) -> tuple[pd.DataFrame, float, float]:
    rows = []
    groups = []
    for cluster in CLUSTERS:
        values = cells.loc[
            cells["epithelial_cluster"].eq(cluster), "signaling_entropy_rate"
        ].to_numpy(dtype=float)
        groups.append(values)
        q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
        iqr = q3 - q1
        rows.append(
            {
                "epithelial_cluster": cluster,
                "n_cells": len(values),
                "n_patients": int(
                    cells.loc[cells["epithelial_cluster"].eq(cluster), "patient_id"].nunique()
                ),
                "mean_SR": float(values.mean()),
                "sd_SR": float(values.std(ddof=1)),
                "q1_SR": float(q1),
                "median_SR": float(median),
                "q3_SR": float(q3),
                "lower_whisker_SR": float(values[values >= q1 - 1.5 * iqr].min()),
                "upper_whisker_SR": float(values[values <= q3 + 1.5 * iqr].max()),
            }
        )
    test = kruskal(*groups)
    summary = pd.DataFrame(rows)
    summary["mean_rank_desc"] = summary["mean_SR"].rank(ascending=False, method="min").astype(int)
    summary["median_rank_desc"] = summary["median_SR"].rank(ascending=False, method="min").astype(int)
    return summary, float(test.statistic), float(test.pvalue)


def plot_figure(cells: pd.DataFrame, summary: pd.DataFrame) -> None:
    data = [
        cells.loc[cells["epithelial_cluster"].eq(cluster), "signaling_entropy_rate"].to_numpy()
        for cluster in CLUSTERS
    ]
    ymin = np.floor((cells["signaling_entropy_rate"].min() - 0.005) / 0.02) * 0.02
    ymax = np.ceil((cells["signaling_entropy_rate"].max() + 0.005) / 0.02) * 0.02

    fig = plt.figure(figsize=(5.35, 3.85), facecolor="white")
    ax = fig.add_axes([0.165, 0.205, 0.775, 0.650])
    boxes = ax.boxplot(
        data,
        positions=np.arange(len(CLUSTERS)),
        widths=0.70,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        boxprops={"linewidth": 1.25, "edgecolor": "#111111"},
        whiskerprops={"linewidth": 1.10, "color": "#111111"},
        capprops={"linewidth": 1.10, "color": "#111111"},
        medianprops={"linewidth": 1.65, "color": "white"},
    )
    for box, cluster in zip(boxes["boxes"], CLUSTERS):
        box.set_facecolor(PALETTE[cluster])

    ax.set_xlim(-0.55, len(CLUSTERS) - 0.45)
    ax.set_ylim(ymin, ymax)
    ax.set_xticks(np.arange(len(CLUSTERS)))
    ax.set_xticklabels(CLUSTERS, rotation=45, ha="right", fontsize=9.0)
    yticks = np.arange(ymin, ymax + 1e-9, 0.03)
    ax.set_yticks(yticks)
    ax.set_yticklabels([f"{value:.2f}" for value in yticks], fontsize=8.3)
    ax.set_xlabel("celltype", fontsize=10.0, labelpad=5)
    ax.set_ylabel("SR Value", fontsize=10.0, labelpad=7)
    ax.set_title("Stemness score", fontsize=10.2, pad=6)
    ax.tick_params(axis="both", width=0.9, length=3.2)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#111111")

    fig.text(0.018, 0.970, "I", fontsize=27, fontweight="normal", va="top", ha="left")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def export(summary: pd.DataFrame, statistic: float, pvalue: float) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(
        RESULT_DIR / "Fig3I_epithelial_SR_distribution_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    highest_mean = summary.loc[summary["mean_SR"].idxmax()]
    highest_median = summary.loc[summary["median_SR"].idxmax()]
    metadata = {
        "figure3g_cell_table": str(FIG3G_CELLS),
        "n_cells": int(summary["n_cells"].sum()),
        "clusters": CLUSTERS,
        "highest_mean_SR_cluster": str(highest_mean["epithelial_cluster"]),
        "highest_mean_SR": float(highest_mean["mean_SR"]),
        "highest_median_SR_cluster": str(highest_median["epithelial_cluster"]),
        "highest_median_SR": float(highest_median["median_SR"]),
        "cell_level_kruskal_wallis_H": statistic,
        "cell_level_kruskal_wallis_p": pvalue,
        "boxplot": "Tukey 1.5 IQR whiskers; outliers hidden to match reference layout",
        "visual_reference": "s41746-025-02297-4 Figure 3I",
        "user_decision": "C5 and C8 retained without exclusion",
        "interpretation_boundary": (
            "The cell-level Kruskal-Wallis test is descriptive because cells are nested within "
            "patients. SR is strongly associated with sequencing complexity. C5 is mixed and C8 "
            "is immune-like; the boxes do not establish lineage or intrinsic stemness."
        ),
    }
    (RESULT_DIR / "Fig3I_epithelial_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    print("Loading Figure 3G SR values...", flush=True)
    cells = load_data()
    summary, statistic, pvalue = summarize(cells)
    print(summary[["epithelial_cluster", "n_cells", "mean_SR", "median_SR"]].to_string(index=False))
    export(summary, statistic, pvalue)
    plot_figure(cells, summary)
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}", flush=True)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
