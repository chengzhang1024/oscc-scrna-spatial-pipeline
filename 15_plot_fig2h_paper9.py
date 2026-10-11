"""Plot paper9 Figure 2H matrices in the stable system Python environment."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results/paper9"
FIGURE = ROOT / "figures/paper9"
PDF_FIGURE = FIGURE / "pdf"


def clean_term(term):
    for prefix in ("HALLMARK_", "KEGG_"):
        term = term.replace(prefix, "")
    return term.replace("_", " ").title()


def reconstruct_fdr(matrix, enrichment_path):
    long = pd.read_csv(enrichment_path)
    output = pd.DataFrame(1.0, index=matrix.index, columns=matrix.columns)
    for _, row in long[long["term"].isin(matrix.index)].iterrows():
        current = output.loc[row["term"], row["cell_type"]]
        output.loc[row["term"], row["cell_type"]] = min(current, float(row["fdr"]))
    return output


def plot(matrix_path, enrichment_path, title, stem, low_power_types):
    matrix = pd.read_csv(matrix_path, index_col=0)
    fdr = reconstruct_fdr(matrix, enrichment_path)
    height = max(7.5, 0.27 * len(matrix) + 2)
    width = max(10.5, 1.05 * len(matrix.columns) + 3)
    fig, ax = plt.subplots(figsize=(width, height))
    values = np.clip(matrix.to_numpy(), -6, 6)
    cmap = LinearSegmentedColormap.from_list("direction", ["#3B78B5", "#F7F7F7", "#D95F59"])
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-6, vmax=6, interpolation="nearest")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([
        cell_type.replace(" ", "\n") + ("\n(low power)" if cell_type in low_power_types else "")
        for cell_type in matrix.columns
    ], fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([clean_term(term) for term in matrix.index], fontsize=7)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            adjusted = fdr.iloc[row, column]
            marker = "**" if adjusted < 0.05 else "*" if adjusted < 0.10 else ""
            if marker:
                ax.text(column, row, marker, ha="center", va="center", color="white", fontsize=7, weight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Signed -log10(FDR): OSCC vs Normal", fontsize=8)
    ax.set_title(title, fontsize=11, weight="bold")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        output_dir = PDF_FIGURE if extension == "pdf" else FIGURE
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    summary = pd.read_csv(RESULT / "fig2h_literature_v3_de_summary.csv")
    low_power = summary.loc[
        summary["up_selection"].str.startswith("nominal")
        | summary["down_selection"].str.startswith("nominal"),
        "cell_type",
    ].tolist()
    plot(
        RESULT / "fig2h_literature_v3_KEGG_metabolic_matrix.csv",
        RESULT / "fig2h_literature_v3_KEGG_metabolic_enrichment.csv",
        "H1  Metabolic pathway reprogramming (GSE181919 paper9 pseudobulk)",
        "Fig2H1_paper9_KEGG_metabolic_pseudobulk",
        low_power,
    )
    plot(
        RESULT / "fig2h_literature_v3_Hallmark_matrix.csv",
        RESULT / "fig2h_literature_v3_Hallmark_enrichment.csv",
        "H2  Hallmark pathway reprogramming (GSE181919 paper9 pseudobulk)",
        "Fig2H2_paper9_Hallmark_pseudobulk",
        low_power,
    )


if __name__ == "__main__":
    main()
