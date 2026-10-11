"""Build epithelial Figure 3J: GO enrichment of the highest-SR cluster markers."""

from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact


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
H5AD = ROOT / "results/integrated_3ds_annotated_paper9.h5ad"
FIG3H_SUMMARY = ROOT / "results/paper9/figure3_epithelial/Fig3H_epithelial_mean_SR_and_sensitivity.csv"
MARKERS = ROOT / "results/paper9/figure3_epithelial/Fig3D_epithelial_cluster_markers.csv.gz"
GMT = ROOT / "results/paper9/gene_sets/Enrichr.GO_Biological_Process_2023.gmt"
RESULT_DIR = ROOT / "results/paper9/figure3_epithelial"
FIGURE_DIR = ROOT / "figures/paper9/figure3_epithelial"
OUTPUT_STEM = FIGURE_DIR / "Fig3J_epithelial_high_SR_cluster_GO_reference_style"
TOP_N = 10
MIN_OVERLAP = 2
BAR_COLOR = "#7B4AA8"


def bh_adjust(pvalues: np.ndarray) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    result = np.empty_like(adjusted)
    result[order] = np.minimum(adjusted, 1.0)
    return result


def parse_gmt() -> tuple[list[tuple[str, set[str]]], set[str]]:
    pathways = []
    universe = set()
    with GMT.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 4:
                continue
            genes = {gene for gene in fields[2:] if gene}
            pathways.append((fields[0], genes))
            universe.update(genes)
    return pathways, universe


def clean_term(term: str) -> str:
    label = re.sub(r"\s*\(GO:\d+\)\s*$", "", term).strip()
    return label[:1].lower() + label[1:] if label else label


def run_enrichment() -> tuple[pd.DataFrame, str, list[str], int]:
    sr_summary = pd.read_csv(FIG3H_SUMMARY)
    target_cluster = str(sr_summary.loc[sr_summary["mean_SR"].idxmax(), "epithelial_cluster"])
    markers = pd.read_csv(MARKERS)
    selected = markers.loc[
        markers["epithelial_cluster"].astype(str).eq(target_cluster)
        & markers["rank"].le(50)
        & markers["mean_difference"].gt(0)
        & markers["pct_in"].ge(0.10)
    ].sort_values("rank")
    query_order = selected["gene"].astype(str).tolist()

    pathways, gmt_universe = parse_gmt()
    backed = ad.read_h5ad(H5AD, backed="r")
    expressed = set(backed.var_names.astype(str))
    backed.file.close()
    universe = expressed & gmt_universe
    query = set(query_order) & universe
    if len(query) < 10:
        raise RuntimeError("Too few target-cluster marker genes overlap the GO universe")

    rows = []
    for term, genes in pathways:
        genes = genes & universe
        overlap = query & genes
        if len(overlap) < MIN_OVERLAP:
            continue
        a = len(overlap)
        b = len(query) - a
        c = len(genes) - a
        d = len(universe) - a - b - c
        odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append(
            {
                "target_cluster": target_cluster,
                "term": term,
                "display_term": clean_term(term),
                "overlap_count": a,
                "query_size": len(query),
                "pathway_size": len(genes),
                "universe_size": len(universe),
                "odds_ratio": float(odds_ratio),
                "pvalue": float(pvalue),
                "overlap_genes": ";".join(sorted(overlap)),
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        raise RuntimeError("No GO terms pass the minimum overlap threshold")
    result["fdr_bh"] = bh_adjust(result["pvalue"].to_numpy())
    result["minus_log10_fdr"] = -np.log10(np.maximum(result["fdr_bh"], 1e-300))
    result["significant_fdr_lt_0.05"] = result["fdr_bh"].lt(0.05)
    result = result.sort_values(
        ["fdr_bh", "pvalue", "overlap_count", "term"],
        ascending=[True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    result["rank"] = np.arange(1, len(result) + 1)
    return result, target_cluster, query_order, len(universe)


def plot_figure(result: pd.DataFrame, target_cluster: str) -> None:
    display = result.head(TOP_N).iloc[::-1].copy()
    maximum = max(float(display["minus_log10_fdr"].max()), 0.5)
    labels = [textwrap.fill(term, width=48) for term in display["display_term"]]
    values = display["minus_log10_fdr"].to_numpy(dtype=float)

    fig_height = max(5.2, 0.53 * len(display) + 1.65)
    fig = plt.figure(figsize=(8.20, fig_height), facecolor="white")
    ax = fig.add_axes([0.455, 0.135, 0.385, 0.775])
    y = np.arange(len(display))
    ax.barh(y, values, color=BAR_COLOR, edgecolor=BAR_COLOR, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7.8, linespacing=0.92)
    ax.set_xlim(0, maximum * 1.10)
    ax.set_xlabel("-log10(adj p-value)", fontsize=9.4, labelpad=4)
    ax.tick_params(axis="x", labelsize=8.0, width=0.7, length=3)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.set_title(f"{target_cluster}-GO enrichment", fontsize=13.0, fontstyle="italic", fontweight="bold", pad=8)
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.legend(
        handles=[Patch(facecolor=BAR_COLOR, edgecolor=BAR_COLOR, label="FDR-ranked")],
        title="GO BP",
        loc="center left",
        bbox_to_anchor=(1.015, 0.72),
        frameon=False,
        fontsize=7.4,
        title_fontsize=7.6,
        handlelength=0.9,
    )
    fig.text(0.018, 0.970, "J", fontsize=27, fontweight="normal", va="top", ha="left")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def export(result: pd.DataFrame, target_cluster: str, query_order: list[str], universe_size: int) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(
        RESULT_DIR / "Fig3J_epithelial_high_SR_cluster_GO_enrichment_all.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        {"target_cluster": target_cluster, "marker_rank": np.arange(1, len(query_order) + 1), "gene": query_order}
    ).to_csv(
        RESULT_DIR / "Fig3J_epithelial_high_SR_cluster_marker_query.csv",
        index=False,
        encoding="utf-8-sig",
    )
    metadata = {
        "target_selection": "cluster with highest cell-level mean SR in Figure 3H",
        "target_cluster": target_cluster,
        "marker_source": str(MARKERS),
        "marker_rule": "Figure 3D top-50 positive markers with pct_in >= 0.10",
        "query_genes_before_universe_intersection": len(query_order),
        "GO_database": str(GMT),
        "universe": "integrated-object genes intersect GO Biological Process 2023 GMT",
        "universe_size": universe_size,
        "test": "one-sided Fisher exact overrepresentation analysis",
        "multiple_testing": "Benjamini-Hochberg over all tested terms with overlap >= 2",
        "n_tested_terms": int(len(result)),
        "n_fdr_lt_0.05": int(result["significant_fdr_lt_0.05"].sum()),
        "display": "top 10 terms by BH FDR; purple horizontal bars",
        "visual_reference": "s41746-025-02297-4 Figure 3J",
        "interpretation_boundary": (
            "The target is C1 because C1, not the reference paper's C4, has the highest mean SR here. "
            "Enrichment uses a selected marker list and is descriptive; terms with FDR >= 0.05 must not "
            "be called statistically significant."
        ),
    }
    (RESULT_DIR / "Fig3J_epithelial_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    print("Running GO BP enrichment for the highest-SR epithelial cluster...", flush=True)
    result, target_cluster, query_order, universe_size = run_enrichment()
    print(
        result.head(TOP_N)[["term", "overlap_count", "odds_ratio", "pvalue", "fdr_bh"]].to_string(index=False),
        flush=True,
    )
    export(result, target_cluster, query_order, universe_size)
    plot_figure(result, target_cluster)
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}", flush=True)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
