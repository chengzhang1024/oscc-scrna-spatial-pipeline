"""Plot Figure 1H as article-style paired pentagonal GO radar charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import pandas as pd


matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
    }
)

ROOT = Path(__file__).resolve().parent
INPUT = ROOT / "results/paper9/Fig1G_epithelial_GO_BP_enriched_pathways.csv"
SELECTED_OUTPUT = ROOT / "results/paper9/Fig1H_selected_GO_pathways.csv"
OUTPUT = ROOT / "figures/paper9/Fig1H_epithelial_GO_radar"

TEAL = "#65BBA0"
GRID = "#BFC2C2"
THRESHOLD = "#8D9191"
INK = "#222222"
RADAR_MAX = 10.0
GRID_LEVELS = [2, 4, 6, 8, 10]

SELECTIONS = {
    "OLK vs Normal": [
        ("Epidermis Development (GO:0008544)", "Epidermis development"),
        ("Keratinocyte Differentiation (GO:0030216)", "Keratinocyte\ndifferentiation"),
        ("Mitochondrial Translation (GO:0032543)", "Mitochondrial translation"),
        ("Monocarboxylic Acid Transport (GO:0015718)", "Monocarboxylic acid\ntransport"),
        (
            "Regulation Of Type II Interferon Production (GO:0032649)",
            "Type II interferon\nproduction",
        ),
    ],
    "OSCC vs Normal": [
        ("Epidermis Development (GO:0008544)", "Epidermis development"),
        ("DNA-templated DNA Replication (GO:0006261)", "DNA replication"),
        ("Glycolytic Process (GO:0006096)", "Glycolytic process"),
        (
            "Regulation Of Interleukin-1 Beta Production (GO:0032651)",
            "Interleukin-1 beta\nproduction",
        ),
        ("Mitochondrial Translation (GO:0032543)", "Mitochondrial translation"),
    ],
}


def unit_vertices() -> np.ndarray:
    angles = np.pi / 2 - np.arange(5) * 2 * np.pi / 5
    return np.column_stack([np.cos(angles), np.sin(angles)])


def load_selected() -> pd.DataFrame:
    enriched = pd.read_csv(INPUT)
    rows = []
    for comparison, selections in SELECTIONS.items():
        subset = enriched[(enriched["comparison"] == comparison) & (enriched["direction"] == "Up")]
        for order, (term, display_label) in enumerate(selections, start=1):
            match = subset[subset["term"] == term]
            if len(match) != 1:
                raise ValueError(f"Expected one row for {comparison}: {term}; found {len(match)}")
            row = match.iloc[0].to_dict()
            row["display_order"] = order
            row["display_label"] = display_label.replace("\n", " ")
            row["enrichment_score"] = -np.log10(max(float(row["fdr"]), np.finfo(float).tiny))
            rows.append(row)
    selected = pd.DataFrame(rows)
    selected.to_csv(SELECTED_OUTPUT, index=False, encoding="utf-8-sig")
    return selected


def draw_radar(ax: plt.Axes, title: str, labels: list[str], values: np.ndarray) -> None:
    vertices = unit_vertices()
    ax.set_aspect("equal")
    ax.set_xlim(-1.50, 1.50)
    ax.set_ylim(-1.38, 1.42)
    ax.axis("off")

    for level in GRID_LEVELS:
        polygon = vertices * (level / RADAR_MAX)
        ax.add_patch(
            Polygon(polygon, closed=True, fill=False, edgecolor=GRID, linewidth=0.85, zorder=0)
        )
    for vertex in vertices:
        ax.plot([0, vertex[0]], [0, vertex[1]], color=GRID, linewidth=0.7, zorder=0)

    significance_radius = -np.log10(0.05) / RADAR_MAX
    ax.add_patch(
        Polygon(
            vertices * significance_radius,
            closed=True,
            fill=False,
            edgecolor=THRESHOLD,
            linewidth=1.0,
            linestyle=(0, (4, 3)),
            zorder=1,
        )
    )

    clipped = np.minimum(values, RADAR_MAX)
    data_polygon = vertices * (clipped[:, None] / RADAR_MAX)
    ax.add_patch(
        Polygon(
            data_polygon,
            closed=True,
            facecolor=TEAL,
            edgecolor=TEAL,
            linewidth=1.8,
            alpha=0.48,
            zorder=2,
        )
    )
    ax.plot(
        np.r_[data_polygon[:, 0], data_polygon[0, 0]],
        np.r_[data_polygon[:, 1], data_polygon[0, 1]],
        color=TEAL,
        linewidth=1.6,
        zorder=3,
    )

    ax.text(0, 1.34, title, ha="center", va="bottom", fontsize=12, weight="bold", fontstyle="italic")
    label_vertices = vertices * 1.17
    rotations = [0, 90, 0, 0, 90]
    alignments = ["center", "center", "center", "center", "center"]
    vertical_alignments = ["bottom", "center", "top", "top", "center"]
    for index, label in enumerate(labels):
        ax.text(
            label_vertices[index, 0],
            label_vertices[index, 1],
            label,
            ha=alignments[index],
            va=vertical_alignments[index],
            rotation=rotations[index],
            rotation_mode="anchor",
            fontsize=8.5,
            color=INK,
            linespacing=1.0,
        )

    for level in [0, 2, 4, 6, 8, 10]:
        y = level / RADAR_MAX
        ax.text(-0.035, y, str(level), ha="right", va="center", fontsize=6.5, color="#777777")

    legend_handles = [
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=5,
            markerfacecolor=THRESHOLD,
            markeredgecolor="none",
            label="FDR < 0.05",
        ),
        Line2D(
            [],
            [],
            marker="s",
            linestyle="none",
            markersize=5,
            markerfacecolor=TEAL,
            markeredgecolor="none",
            label="enrichment score (-log10 FDR)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.13),
        frameon=False,
        fontsize=6.3,
        ncol=1,
        handlelength=0.8,
        handletextpad=0.35,
        borderaxespad=0,
        labelspacing=0.25,
    )


def main() -> None:
    selected = load_selected()
    figure = plt.figure(figsize=(8.8, 4.35), facecolor="white")
    left_ax = figure.add_axes([0.075, 0.15, 0.39, 0.73])
    right_ax = figure.add_axes([0.535, 0.15, 0.39, 0.73])

    panels = [
        (left_ax, "OLK — Top 5 GO", "OLK vs Normal"),
        (right_ax, "OSCC — Top 5 GO", "OSCC vs Normal"),
    ]
    for axis, title, comparison in panels:
        panel = selected[selected["comparison"] == comparison].sort_values("display_order")
        labels = [display.replace(" ", " ") for display in panel["display_label"]]
        # Restore intentional wrapping used by the article-style peripheral labels.
        labels = [
            dict(SELECTIONS[comparison])[term]
            for term in panel["term"]
        ]
        draw_radar(axis, title, labels, panel["enrichment_score"].to_numpy(dtype=float))

    figure.text(0.018, 0.955, "H", fontsize=24, ha="left", va="top")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT.with_suffix(".png"), dpi=400, facecolor="white")
    figure.savefig(OUTPUT.with_suffix(".pdf"), dpi=400, facecolor="white")
    plt.close(figure)
    print(f"Saved: {OUTPUT.with_suffix('.png')}")
    print(f"Saved: {OUTPUT.with_suffix('.pdf')}")
    print(f"Saved: {SELECTED_OUTPUT}")


if __name__ == "__main__":
    main()
