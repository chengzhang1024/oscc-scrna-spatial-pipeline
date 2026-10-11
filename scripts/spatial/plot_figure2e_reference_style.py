"""Plot Figure 2E from the patient-aggregated Early-stage MISTy intra view."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib

matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }
)
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.patheffects as path_effects
from matplotlib.patches import Ellipse, FancyArrowPatch, Polygon, Rectangle
import networkx as nx
import numpy as np
import pandas as pd


INPUT = ROOT / "results" / "spatial_misty" / "figure2c_early" / "early_misty_group_importance.csv"
SAMPLE_INPUT = (
    ROOT
    / "results"
    / "spatial_misty"
    / "figure2c_early"
    / "early_misty_interactions_by_sample.csv"
)
OUTPUT_DIR = ROOT / "figures" / "paper9" / "figure2"
OUTPUT = OUTPUT_DIR / "Fig2E_MISTy_intra_network_early_OSCC"
EDGE_AUDIT = OUTPUT_DIR / "Fig2E_MISTy_intra_network_edges.csv"
NODE_AUDIT = OUTPUT_DIR / "Fig2E_MISTy_intra_network_nodes.csv"
PANEL_LETTER = "E"
STABLE_TIER_LABEL = "stable_3of4_or_more"
CANDIDATE_TIER_LABEL = "candidate_2of4"
EDGE_NOTE = (
    "Intra-view median importance z >= 1; solid edges have z >= 1 in >=3/4 "
    "patients, dashed edges in 2/4 patients; arrow points predictor to target; "
    "background modules use weighted Louvain community detection (seed 181919)"
)

CELL_TYPES = [
    "B/Plasma cells",
    "Dendritic cells",
    "Endothelial cells",
    "Epithelial cells",
    "Fibroblasts",
    "Macrophages",
    "Mast cells",
    "Myocytes",
    "NK/T cells",
]

LABELS = {
    "B/Plasma cells": "B/Plasma",
    "Dendritic cells": "D_C",
    "Endothelial cells": "Endothelial",
    "Epithelial cells": "Epithelial",
    "Fibroblasts": "Fibro",
    "Macrophages": "Macrophage",
    "Mast cells": "Mast",
    "Myocytes": "Myocyte",
    "NK/T cells": "T_NK",
}

CELL_COLORS = {
    "B/Plasma cells": "#65BBA0",
    "Dendritic cells": "#F9DF95",
    "Endothelial cells": "#8AC9BE",
    "Epithelial cells": "#CF535E",
    "Fibroblasts": "#F6B273",
    "Macrophages": "#5CA48E",
    "Mast cells": "#D3304F",
    "Myocytes": "#E76C43",
    "NK/T cells": "#F7B52B",
}

# Fixed coordinates follow the reference Figure 2E's narrow, vertically layered
# network while keeping the patient-aggregated directional edges readable.
POSITIONS = {
    "Mast cells": (-0.68, 0.72),
    "B/Plasma cells": (-0.18, 0.82),
    "Dendritic cells": (0.58, 0.67),
    "NK/T cells": (0.32, 0.55),
    "Epithelial cells": (0.00, 0.21),
    "Fibroblasts": (-0.35, -0.02),
    "Endothelial cells": (-0.55, -0.39),
    "Myocytes": (0.28, -0.05),
    "Macrophages": (0.52, -0.40),
}

LABEL_OFFSETS = {
    "Mast cells": (-0.03, 0.075, "center"),
    "B/Plasma cells": (0.00, 0.075, "center"),
    "Dendritic cells": (0.03, 0.072, "center"),
    "NK/T cells": (-0.02, 0.070, "center"),
    "Epithelial cells": (-0.08, 0.065, "right"),
    "Fibroblasts": (-0.04, -0.072, "center"),
    "Endothelial cells": (0.00, -0.075, "center"),
    "Myocytes": (0.02, -0.073, "center"),
    "Macrophages": (0.00, -0.076, "center"),
}

STRONG_STABILITY_MIN = 0.75
CANDIDATE_STABILITY_MIN = 0.50
IMPORTANCE_MIN = 1.00
COMMUNITY_SEED = 181919
RECIPROCAL_CURVATURE = 0.20

COMMUNITY_STYLES = {
    "epithelial_stromal": {"fill": "#F3A4B2", "edge": "#D75A74"},
    "lymphoid_dendritic": {"fill": "#F3D46C", "edge": "#C89D20"},
    "myocyte_macrophage": {"fill": "#82D2B5", "edge": "#36A17D"},
}


def darken(hex_color: str, factor: float = 0.67) -> tuple[float, float, float]:
    return tuple(np.clip(np.asarray(colors.to_rgb(hex_color)) * factor, 0, 1))


def selected_directional_edges(data: pd.DataFrame) -> pd.DataFrame:
    selected = data.loc[
        data["view"].eq("intra")
        & data["stability_fraction"].ge(CANDIDATE_STABILITY_MIN)
        & data["median_importance_z"].ge(IMPORTANCE_MIN)
    ].copy()
    if selected.empty:
        raise RuntimeError("No intra-view relationships passed the Figure 2E display thresholds")
    selected["evidence_tier"] = np.where(
        selected["stability_fraction"].ge(STRONG_STABILITY_MIN),
        STABLE_TIER_LABEL,
        CANDIDATE_TIER_LABEL,
    )
    return selected.sort_values("median_importance_z", ascending=False)


def directional_graph(edges: pd.DataFrame) -> nx.DiGraph:
    graph = nx.DiGraph()
    graph.add_nodes_from(CELL_TYPES)
    for row in edges.itertuples(index=False):
        graph.add_edge(
            str(row.predictor),
            str(row.target),
            weight=float(row.median_importance_z),
            evidence_tier=str(row.evidence_tier),
        )
    return graph


def detect_communities(edges: pd.DataFrame) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Detect weighted topology modules; isolated nodes are not shaded."""
    graph = nx.Graph()
    graph.add_nodes_from(CELL_TYPES)
    for row in edges.itertuples(index=False):
        predictor = str(row.predictor)
        target = str(row.target)
        previous = float(graph.get_edge_data(predictor, target, {}).get("weight", 0.0))
        graph.add_edge(
            predictor,
            target,
            weight=previous + float(row.median_importance_z),
        )

    communities = nx.community.louvain_communities(
        graph,
        weight="weight",
        resolution=1.0,
        seed=COMMUNITY_SEED,
    )
    membership: dict[str, str] = {}
    named: dict[str, set[str]] = {}
    for members in communities:
        members = set(members)
        if len(members) == 1:
            membership[next(iter(members))] = "isolated"
            continue
        if "Epithelial cells" in members:
            module = "epithelial_stromal"
        elif "NK/T cells" in members:
            module = "lymphoid_dendritic"
        elif "Myocytes" in members:
            module = "myocyte_macrophage"
        else:
            module = f"module_{len(named) + 1}"
        named[module] = members
        membership.update({cell_type: module for cell_type in members})
    return membership, named


def convex_hull(points: np.ndarray) -> np.ndarray:
    """Return a 2D monotonic-chain convex hull in counter-clockwise order."""
    ordered = sorted({(float(x), float(y)) for x, y in points})
    if len(ordered) <= 2:
        return np.asarray(ordered)

    def cross(origin, a, b):
        return (a[0] - origin[0]) * (b[1] - origin[1]) - (
            a[1] - origin[1]
        ) * (b[0] - origin[0])

    lower = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1])


def draw_community_envelope(ax: plt.Axes, members: set[str], style: dict[str, str]) -> None:
    points = np.asarray([POSITIONS[cell_type] for cell_type in members], dtype=float)
    if len(points) == 2:
        start, end = points
        delta = end - start
        center = (start + end) / 2
        patch = Ellipse(
            center,
            width=float(np.linalg.norm(delta)) + 0.20,
            height=0.17,
            angle=float(np.degrees(np.arctan2(delta[1], delta[0]))),
            facecolor=style["fill"],
            edgecolor=style["edge"],
            linewidth=0.65,
            alpha=0.34,
            zorder=0,
        )
    else:
        hull = convex_hull(points)
        center = hull.mean(axis=0)
        vectors = hull - center
        lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
        expanded = hull + vectors / np.maximum(lengths, 1e-9) * 0.10
        patch = Polygon(
            expanded,
            closed=True,
            facecolor=style["fill"],
            edgecolor=style["edge"],
            linewidth=0.65,
            joinstyle="round",
            alpha=0.30,
            zorder=0,
        )
    ax.add_patch(patch)


def add_patient_support(edges: pd.DataFrame, by_sample: pd.DataFrame) -> pd.DataFrame:
    support = by_sample.loc[
        by_sample["view"].eq("intra") & by_sample["importance_z"].ge(IMPORTANCE_MIN)
    ].copy()
    if "sample_id" in support.columns:
        support["patient_support"] = support.apply(
            lambda row: f"{row['patient_id']}({row['sample_id']}):z={row['importance_z']:.2f}",
            axis=1,
        )
    else:
        support["patient_support"] = support.apply(
            lambda row: f"{row['patient_id']}:z={row['importance_z']:.2f}", axis=1
        )
    grouped = (
        support.sort_values(["target", "predictor", "patient_id"])
        .groupby(["target", "predictor"], as_index=False)
        .agg(
            supporting_patients=(
                "patient_id", lambda values: ";".join(map(str, pd.unique(values)))
            ),
            patient_support=("patient_support", "; ".join),
        )
    )
    return edges.merge(grouped, on=["target", "predictor"], how="left")


def main() -> None:
    data = pd.read_csv(INPUT)
    by_sample = pd.read_csv(SAMPLE_INPUT)
    directional = add_patient_support(selected_directional_edges(data), by_sample)
    graph = directional_graph(directional)
    membership, communities = detect_communities(directional)

    # The reference panel is a compact portrait panel embedded between D and F.
    fig = plt.figure(figsize=(1.72, 2.18), facecolor="white")
    ax = fig.add_axes([0.075, 0.075, 0.89, 0.79])
    ax.set_xlim(-0.86, 0.79)
    ax.set_ylim(-0.61, 1.00)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    for module, members in communities.items():
        style = COMMUNITY_STYLES.get(
            module,
            {"fill": "#A7B8D8", "edge": "#60779C"},
        )
        draw_community_envelope(ax, members, style)

    min_importance = float(directional["median_importance_z"].min())
    max_importance = float(directional["median_importance_z"].max())
    importance_span = max(max_importance - min_importance, 1e-6)
    pairs = directional.apply(
        lambda row: tuple(sorted((str(row["predictor"]), str(row["target"])))), axis=1
    )
    reciprocal_pairs = set(pairs[pairs.duplicated(keep=False)])
    for row in directional.itertuples(index=False):
        predictor = str(row.predictor)
        target = str(row.target)
        pair = tuple(sorted((predictor, target)))
        if pair in reciprocal_pairs:
            # Reversing the endpoints already reverses the absolute bend side.
            # Keeping the same radial sign separates reciprocal arrows.
            curvature = RECIPROCAL_CURVATURE
        else:
            curvature = 0.0
        normalized = (float(row.median_importance_z) - min_importance) / importance_span
        stable = float(row.stability_fraction) >= STRONG_STABILITY_MIN
        linewidth = (1.05 + 0.55 * normalized) if stable else (0.72 + 0.38 * normalized)
        arrow = FancyArrowPatch(
            POSITIONS[predictor],
            POSITIONS[target],
            arrowstyle="-|>",
            mutation_scale=7.2,
            connectionstyle=f"arc3,rad={curvature}",
            color="#CF535E" if stable else "#65BBA0",
            alpha=0.98 if stable else 0.92,
            linewidth=linewidth,
            linestyle="solid" if stable else (0, (3.0, 2.2)),
            shrinkA=4.4,
            shrinkB=4.4,
            zorder=2,
        )
        arrow.set_path_effects(
            [
                path_effects.Stroke(
                    linewidth=linewidth + 0.85,
                    foreground="white",
                    alpha=0.78,
                ),
                path_effects.Normal(),
            ]
        )
        ax.add_patch(arrow)

    for cell_type in CELL_TYPES:
        x, y = POSITIONS[cell_type]
        ax.scatter(
            [x],
            [y],
            s=11.5,
            c="#111111",
            edgecolors="#111111",
            linewidths=0.20,
            zorder=4,
        )
        dx, dy, alignment = LABEL_OFFSETS[cell_type]
        ax.text(
            x + dx,
            y + dy,
            LABELS[cell_type],
            ha=alignment,
            va="center",
            fontsize=5.05,
            fontweight="bold",
            color=darken(CELL_COLORS[cell_type], factor=0.72),
            zorder=5,
        )

    strip = Rectangle(
        (0.115, 0.902),
        0.865,
        0.067,
        transform=fig.transFigure,
        facecolor="#D9D9DD",
        edgecolor="none",
        clip_on=False,
        zorder=10,
    )
    fig.add_artist(strip)
    fig.text(
        0.548,
        0.936,
        "Misty-intra",
        ha="center",
        va="center",
        fontsize=7.6,
        fontweight="bold",
        fontstyle="italic",
        color="#202020",
        zorder=11,
    )
    fig.text(
        0.012, 0.982, PANEL_LETTER, ha="left", va="top", fontsize=10.6, color="#202020"
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    directional.assign(
        display_mode="directed",
        predictor_module=directional["predictor"].map(membership),
        target_module=directional["target"].map(membership),
        note=EDGE_NOTE,
    ).to_csv(EDGE_AUDIT, index=False)
    pd.DataFrame(
        [
            {
                "cell_type": cell_type,
                "label": LABELS[cell_type],
                "x": POSITIONS[cell_type][0],
                "y": POSITIONS[cell_type][1],
                "in_degree": int(graph.in_degree(cell_type)),
                "out_degree": int(graph.out_degree(cell_type)),
                "network_status": "connected" if graph.degree(cell_type) > 0 else "isolated",
                "community": membership[cell_type],
                "community_members": ";".join(
                    sorted(communities.get(membership[cell_type], {cell_type}))
                ),
                "community_method": "weighted_louvain_seed_181919",
            }
            for cell_type in CELL_TYPES
        ]
    ).to_csv(NODE_AUDIT, index=False)
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=600, facecolor="white")
    plt.close(fig)
    n_stable = int(directional["stability_fraction"].ge(STRONG_STABILITY_MIN).sum())
    n_candidate = len(directional) - n_stable
    print(
        f"Saved {len(directional)} directed relationships: "
        f"{n_stable} stable and {n_candidate} candidate"
    )
    print(f"Saved {OUTPUT.with_suffix('.png')}")
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")
    print(f"Saved {EDGE_AUDIT}")
    print(f"Saved {NODE_AUDIT}")


if __name__ == "__main__":
    main()
