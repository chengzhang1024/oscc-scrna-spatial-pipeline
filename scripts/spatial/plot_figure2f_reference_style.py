"""Infer reproducible Early-stage spatial niches and plot Figure 2F."""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import anndata as ad
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
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, leaves_list, linkage
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.neighbors import NearestNeighbors


INPUT = ROOT / "data" / "processed" / "GSE208253" / "GSE208253_cleaned_deconvolved.h5ad"
OUTPUT_DIR = ROOT / "figures" / "paper9" / "figure2"
SELECTION = OUTPUT_DIR / "Fig2A_representative_sample_selection.csv"
OUTPUT = OUTPUT_DIR / "Fig2F_spatial_niches_early_OSCC"
K_AUDIT = OUTPUT_DIR / "Fig2F_niche_k_selection.csv"
ASSIGNMENT_AUDIT = OUTPUT_DIR / "Fig2F_niche_assignments_early_OSCC.csv.gz"
ENRICHMENT_AUDIT = OUTPUT_DIR / "Fig2F_niche_enrichment_early_OSCC.csv"
PATIENT_ENRICHMENT_AUDIT = OUTPUT_DIR / "Fig2F_niche_enrichment_by_patient.csv"
PREVALENCE_AUDIT = OUTPUT_DIR / "Fig2F_niche_patient_prevalence.csv"
METADATA_AUDIT = OUTPUT_DIR / "Fig2F_niche_model_metadata.json"
PANEL_LETTER = "F"
SPATIAL_TITLE = "Niche-early OSCC"
HEATMAP_TITLE = "Abundance-early OSCC by niche"

plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["svg.fonttype"] = "none"

SEED = 181919
NEIGHBORHOOD_SIZE = 6  # center spot plus five nearest spots
TRAIN_PER_PATIENT = 800
K_CANDIDATES = range(3, 9)
BOOTSTRAP_REPEATS = 12
MIN_BOOTSTRAP_ARI = 0.90
MIN_GROUP_FRACTION = 0.05
MIN_PATIENT_FRACTION = 0.01
MIN_PATIENTS_PRESENT = 3
DOT_MIN_ABS_LOG2_ROE = 0.15

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

SHORT_LABELS = {
    "B/Plasma cells": "B/Plasma",
    "Dendritic cells": "Dendritic",
    "Endothelial cells": "Endothelial",
    "Epithelial cells": "Epithelial",
    "Fibroblasts": "Fibro",
    "Macrophages": "Macrophage",
    "Mast cells": "Mast",
    "Myocytes": "Myocytes",
    "NK/T cells": "T_NK",
}

NICHE_COLORS_BY_NAME = {
    "Epithelial-rich": "#CF535E",
    "Fibrovascular": "#65BBA0",
    "Immune-mixed": "#F7B52B",
    "Myocyte-rich": "#E76C43",
}
FALLBACK_NICHE_COLORS = ["#5CA48E", "#D3304F", "#8AC9BE", "#F6B273"]


def padded_limits(values: np.ndarray, fraction: float = 0.025) -> tuple[float, float]:
    lower = float(np.nanmin(values))
    upper = float(np.nanmax(values))
    padding = max((upper - lower) * fraction, 1.0)
    return lower - padding, upper + padding


def build_local_profiles(
    proportions: np.ndarray,
    coordinates: np.ndarray,
    sample_ids: np.ndarray,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    local = np.zeros_like(proportions, dtype=float)
    spatial_graphs: list[tuple[np.ndarray, np.ndarray]] = []
    for sample_id in sorted(set(sample_ids)):
        sample_index = np.flatnonzero(sample_ids == sample_id)
        n_neighbors = min(NEIGHBORHOOD_SIZE, len(sample_index))
        model = NearestNeighbors(n_neighbors=n_neighbors).fit(coordinates[sample_index])
        neighbor_index = model.kneighbors(
            coordinates[sample_index], return_distance=False
        )
        local[sample_index] = proportions[sample_index][neighbor_index].mean(axis=1)
        spatial_graphs.append((sample_index, neighbor_index[:, 1:]))
    return local, spatial_graphs


def patient_centered_clr(local: np.ndarray, patient_ids: np.ndarray) -> np.ndarray:
    transformed = np.log(np.clip(local, 1e-6, None))
    transformed -= transformed.mean(axis=1, keepdims=True)
    for patient_id in sorted(set(patient_ids)):
        selected = patient_ids == patient_id
        transformed[selected] -= np.median(transformed[selected], axis=0)
    return transformed


def balanced_training_index(patient_ids: np.ndarray) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    selected = []
    for patient_id in sorted(set(patient_ids)):
        available = np.flatnonzero(patient_ids == patient_id)
        n_select = min(TRAIN_PER_PATIENT, len(available))
        selected.append(rng.choice(available, size=n_select, replace=False))
    return np.concatenate(selected)


def spatial_agreement_excess(
    labels: np.ndarray,
    spatial_graphs: list[tuple[np.ndarray, np.ndarray]],
    n_clusters: int,
) -> tuple[float, float, float]:
    observed = []
    expected = []
    for sample_index, neighbor_index in spatial_graphs:
        sample_labels = labels[sample_index]
        observed.append(float((sample_labels[:, None] == sample_labels[neighbor_index]).mean()))
        fractions = np.bincount(sample_labels, minlength=n_clusters) / len(sample_labels)
        expected.append(float(np.square(fractions).sum()))
    observed_mean = float(np.mean(observed))
    expected_mean = float(np.mean(expected))
    return observed_mean, expected_mean, observed_mean - expected_mean


def evaluate_k_values(
    features: np.ndarray,
    train_index: np.ndarray,
    patient_ids: np.ndarray,
    sample_ids: np.ndarray,
    spatial_graphs: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[pd.DataFrame, dict[int, tuple[KMeans, np.ndarray]]]:
    eval_rng = np.random.default_rng(SEED + 17)
    eval_index = eval_rng.choice(
        len(features), size=min(3000, len(features)), replace=False
    )
    fitted: dict[int, tuple[KMeans, np.ndarray]] = {}
    rows = []
    for n_clusters in K_CANDIDATES:
        model = KMeans(n_clusters=n_clusters, n_init=50, random_state=SEED)
        model.fit(features[train_index])
        labels = model.predict(features)
        fitted[n_clusters] = (model, labels)

        bootstrap_rng = np.random.default_rng(SEED + n_clusters * 100)
        bootstrap_ari = []
        for repeat in range(BOOTSTRAP_REPEATS):
            bootstrap_index = bootstrap_rng.choice(
                train_index,
                size=int(0.8 * len(train_index)),
                replace=False,
            )
            bootstrap_model = KMeans(
                n_clusters=n_clusters,
                n_init=20,
                random_state=SEED + n_clusters * 1000 + repeat,
            ).fit(features[bootstrap_index])
            bootstrap_ari.append(
                adjusted_rand_score(
                    labels[eval_index], bootstrap_model.predict(features[eval_index])
                )
            )

        cluster_fractions = np.bincount(labels, minlength=n_clusters) / len(labels)
        patients_per_cluster = []
        for cluster in range(n_clusters):
            patients_per_cluster.append(
                sum(
                    float((labels[patient_ids == patient_id] == cluster).mean())
                    >= MIN_PATIENT_FRACTION
                    for patient_id in sorted(set(patient_ids))
                )
            )
        observed, expected, excess = spatial_agreement_excess(
            labels, spatial_graphs, n_clusters
        )
        eligible = (
            float(np.mean(bootstrap_ari)) >= MIN_BOOTSTRAP_ARI
            and float(cluster_fractions.min()) >= MIN_GROUP_FRACTION
            and min(patients_per_cluster) >= MIN_PATIENTS_PRESENT
        )
        rows.append(
            {
                "n_niches": n_clusters,
                "silhouette": silhouette_score(
                    features[eval_index], labels[eval_index]
                ),
                "bootstrap_ari_mean": float(np.mean(bootstrap_ari)),
                "bootstrap_ari_min": float(np.min(bootstrap_ari)),
                "min_group_fraction": float(cluster_fractions.min()),
                "min_patients_present": int(min(patients_per_cluster)),
                "all_niches_in_3of4_or_more": bool(
                    min(patients_per_cluster) >= MIN_PATIENTS_PRESENT
                ),
                "spatial_neighbor_agreement": observed,
                "spatial_neighbor_null": expected,
                "spatial_agreement_excess": excess,
                "sample_nmi": normalized_mutual_info_score(sample_ids, labels),
                "eligible": eligible,
            }
        )
    audit = pd.DataFrame(rows)
    eligible_k = audit.loc[audit["eligible"], "n_niches"]
    if eligible_k.empty:
        raise RuntimeError("No niche resolution passed the prespecified stability criteria")
    selected_k = int(eligible_k.max())
    audit["selected"] = audit["n_niches"].eq(selected_k)
    return audit, fitted


def enrichment_tables(
    local: np.ndarray,
    labels: np.ndarray,
    patient_ids: np.ndarray,
    n_clusters: int,
) -> tuple[np.ndarray, pd.DataFrame, pd.DataFrame]:
    patient_rows = []
    patients = sorted(set(patient_ids))
    for patient_id in patients:
        patient_mask = patient_ids == patient_id
        expected = local[patient_mask].mean(axis=0)
        for cluster in range(n_clusters):
            selected = patient_mask & (labels == cluster)
            if not selected.any():
                values = np.full(len(CELL_TYPES), np.nan)
            else:
                observed = local[selected].mean(axis=0)
                values = np.log2((observed + 1e-6) / (expected + 1e-6))
            for cell_type, value in zip(CELL_TYPES, values):
                patient_rows.append(
                    {
                        "patient_id": patient_id,
                        "cluster": cluster,
                        "cell_type": cell_type,
                        "log2_roe": value,
                        "n_spots": int(selected.sum()),
                    }
                )
    by_patient = pd.DataFrame(patient_rows)
    group_rows = []
    matrix = np.zeros((len(CELL_TYPES), n_clusters), dtype=float)
    for cluster in range(n_clusters):
        for cell_index, cell_type in enumerate(CELL_TYPES):
            values = by_patient.loc[
                by_patient["cluster"].eq(cluster)
                & by_patient["cell_type"].eq(cell_type),
                "log2_roe",
            ].dropna()
            median = float(values.median())
            support = int((values.gt(0) if median >= 0 else values.lt(0)).sum())
            matrix[cell_index, cluster] = median
            group_rows.append(
                {
                    "cluster": cluster,
                    "cell_type": cell_type,
                    "median_log2_roe": median,
                    "n_patients_present": int(len(values)),
                    "n_patients_same_direction": support,
                    "display_dot": bool(
                        support >= MIN_PATIENTS_PRESENT
                        and abs(median) >= DOT_MIN_ABS_LOG2_ROE
                    ),
                }
            )
    return matrix, pd.DataFrame(group_rows), by_patient


def infer_niche_names(matrix: np.ndarray) -> dict[int, str]:
    names = {}
    used: dict[str, int] = {}
    for cluster in range(matrix.shape[1]):
        top_cell = CELL_TYPES[int(np.argmax(matrix[:, cluster]))]
        if top_cell == "Epithelial cells":
            base = "Epithelial-rich"
        elif top_cell in {"Fibroblasts", "Endothelial cells"}:
            base = "Fibrovascular"
        elif top_cell in {
            "B/Plasma cells",
            "Dendritic cells",
            "Macrophages",
            "NK/T cells",
        }:
            base = "Immune-mixed"
        elif top_cell == "Myocytes":
            base = "Myocyte-rich"
        else:
            base = f"{SHORT_LABELS[top_cell]}-rich"
        used[base] = used.get(base, 0) + 1
        names[cluster] = base if used[base] == 1 else f"{base}-{used[base]}"
    return names


def niche_color(name: str, fallback_index: int) -> str:
    base = name.rsplit("-", 1)[0] if name[-1:].isdigit() else name
    return NICHE_COLORS_BY_NAME.get(
        base, FALLBACK_NICHE_COLORS[fallback_index % len(FALLBACK_NICHE_COLORS)]
    )


def plot_figure(
    adata: ad.AnnData,
    early_mask: np.ndarray,
    obs: pd.DataFrame,
    coordinates: np.ndarray,
    labels: np.ndarray,
    matrix: np.ndarray,
    enrichment: pd.DataFrame,
    names: dict[int, str],
    representative_sample: str,
    fixed_cluster_to_id: dict[int, str] | None = None,
    fixed_cluster_to_color: dict[int, str] | None = None,
    ood_mask: np.ndarray | None = None,
) -> tuple[dict[int, str], dict[int, str]]:
    row_linkage = linkage(matrix, method="average", metric="euclidean", optimal_ordering=True)
    col_linkage = linkage(matrix.T, method="average", metric="euclidean", optimal_ordering=True)
    row_order = leaves_list(row_linkage)
    col_order = leaves_list(col_linkage)
    cluster_to_id = fixed_cluster_to_id or {
        int(cluster): f"N{index + 1}" for index, cluster in enumerate(col_order)
    }
    cluster_to_color = fixed_cluster_to_color or {
        int(cluster): niche_color(names[int(cluster)], index)
        for index, cluster in enumerate(col_order)
    }

    fig = plt.figure(figsize=(6.10, 2.45), facecolor="white")
    fig.text(
        0.006, 0.985, PANEL_LETTER, ha="left", va="top", fontsize=12.5, color="#202020"
    )
    for x, width, title in [
        (0.055, 0.425, SPATIAL_TITLE),
        (0.530, 0.455, HEATMAP_TITLE),
    ]:
        fig.add_artist(
            Rectangle(
                (x, 0.895),
                width,
                0.075,
                transform=fig.transFigure,
                facecolor="#D9D9DD",
                edgecolor="none",
                clip_on=False,
                zorder=10,
            )
        )
        fig.text(
            x + width / 2,
            0.932,
            title,
            ha="center",
            va="center",
            fontsize=8.2,
            fontweight="bold",
            fontstyle="italic",
            color="#202020",
            zorder=11,
        )

    spatial_ax = fig.add_axes([0.060, 0.205, 0.410, 0.645])
    representative = obs["sample_id"].astype(str).eq(representative_sample).to_numpy()
    representative_coordinates = coordinates[representative]
    representative_labels = labels[representative]
    representative_ood = (
        np.zeros(len(representative_labels), dtype=bool)
        if ood_mask is None
        else np.asarray(ood_mask, dtype=bool)[representative]
    )
    for cluster in col_order:
        selected = (representative_labels == cluster) & ~representative_ood
        spatial_ax.scatter(
            representative_coordinates[selected, 0],
            representative_coordinates[selected, 1],
            s=4.6,
            c=cluster_to_color[int(cluster)],
            edgecolors="none",
            linewidths=0,
            alpha=0.96,
            rasterized=True,
        )
    if representative_ood.any():
        spatial_ax.scatter(
            representative_coordinates[representative_ood, 0],
            representative_coordinates[representative_ood, 1],
            s=4.6,
            c="#B7B7B7",
            edgecolors="none",
            linewidths=0,
            alpha=0.96,
            rasterized=True,
        )
    spatial_ax.set_xlim(padded_limits(representative_coordinates[:, 0]))
    spatial_ax.set_ylim(padded_limits(representative_coordinates[:, 1])[::-1])
    spatial_ax.set_aspect("equal", adjustable="box")
    spatial_ax.axis("off")

    scalefactors = adata.uns["spatial"][representative_sample]["scalefactors"]
    pixels_per_micron = float(scalefactors["spot_diameter_fullres"]) / 55.0
    scale_fraction = (
        500.0
        * pixels_per_micron
        / (spatial_ax.get_xlim()[1] - spatial_ax.get_xlim()[0])
    )
    scale_x = 0.08
    scale_y = 0.08
    spatial_ax.add_line(
        Line2D(
            [scale_x, scale_x + scale_fraction],
            [scale_y, scale_y],
            transform=spatial_ax.transAxes,
            color="#111111",
            linewidth=1.5,
            solid_capstyle="butt",
        )
    )
    spatial_ax.text(
        scale_x + scale_fraction / 2,
        scale_y + 0.035,
        r"500 $\mu$m",
        transform=spatial_ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=5.0,
        color="#111111",
    )

    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=4.2,
            markerfacecolor=cluster_to_color[int(cluster)],
            markeredgecolor="none",
            label=f"{cluster_to_id[int(cluster)]}  {names[int(cluster)]}",
        )
        for cluster in col_order
    ]
    if representative_ood.any():
        handles.append(
            Line2D(
                [0],
                [0],
                marker="o",
                linestyle="",
                markersize=4.2,
                markerfacecolor="#B7B7B7",
                markeredgecolor="none",
                label="OOD  advanced-deviant",
            )
        )
    fig.legend(
        handles=handles,
        loc="center",
        bbox_to_anchor=(0.265, 0.083),
        ncol=4 if representative_ood.any() else 3,
        frameon=False,
        fontsize=5.0,
        handletextpad=0.35,
        columnspacing=0.9,
        labelspacing=0.25,
    )

    row_dendrogram_ax = fig.add_axes([0.545, 0.205, 0.060, 0.505])
    dendrogram(
        row_linkage,
        orientation="left",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#6A6A6A",
        ax=row_dendrogram_ax,
    )
    row_dendrogram_ax.axis("off")

    column_dendrogram_ax = fig.add_axes([0.610, 0.715, 0.245, 0.135])
    dendrogram(
        col_linkage,
        orientation="top",
        no_labels=True,
        color_threshold=0,
        above_threshold_color="#6A6A6A",
        ax=column_dendrogram_ax,
    )
    column_dendrogram_ax.axis("off")

    heatmap_ax = fig.add_axes([0.610, 0.205, 0.245, 0.505])
    display_matrix = matrix[np.ix_(row_order, col_order)]
    limit = max(0.5, float(np.quantile(np.abs(display_matrix), 0.98)))
    heatmap_cmap = colors.LinearSegmentedColormap.from_list(
        "paper9_niche_enrichment", ["#4E8E91", "#F7F7F7", "#CF535E"], N=256
    )
    image = heatmap_ax.imshow(
        display_matrix,
        cmap=heatmap_cmap,
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
        aspect="auto",
    )
    heatmap_ax.set_xticks(range(len(col_order)))
    heatmap_ax.set_xticklabels(
        [cluster_to_id[int(cluster)] for cluster in col_order], fontsize=5.3
    )
    heatmap_ax.set_yticks(range(len(row_order)))
    heatmap_ax.set_yticklabels(
        [SHORT_LABELS[CELL_TYPES[index]] for index in row_order], fontsize=4.9
    )
    heatmap_ax.yaxis.tick_right()
    heatmap_ax.tick_params(axis="both", length=0, pad=1.5)
    heatmap_ax.set_xticks(np.arange(-0.5, len(col_order), 1), minor=True)
    heatmap_ax.set_yticks(np.arange(-0.5, len(row_order), 1), minor=True)
    heatmap_ax.grid(which="minor", color="white", linewidth=0.75)
    heatmap_ax.tick_params(which="minor", bottom=False, left=False)
    for spine in heatmap_ax.spines.values():
        spine.set_linewidth(0.45)
        spine.set_color("#555555")

    dot_lookup = enrichment.set_index(["cell_type", "cluster"])["display_dot"]
    for display_row, cell_index in enumerate(row_order):
        for display_column, cluster in enumerate(col_order):
            if bool(dot_lookup.loc[(CELL_TYPES[cell_index], int(cluster))]):
                heatmap_ax.scatter(
                    display_column,
                    display_row,
                    s=3.0,
                    c="#252525",
                    edgecolors="none",
                    zorder=5,
                )

    colorbar_ax = fig.add_axes([0.942, 0.325, 0.008, 0.255])
    colorbar = fig.colorbar(image, cax=colorbar_ax)
    colorbar.set_ticks([-limit, 0.0, limit])
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    colorbar.ax.tick_params(labelsize=4.4, length=1.5, width=0.4)
    colorbar.outline.set_linewidth(0.4)
    colorbar.ax.set_title("log2\nRo/e", fontsize=4.3, pad=2.0)

    fig.savefig(OUTPUT.with_suffix(".png"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT.with_suffix(".pdf"), dpi=600, facecolor="white")
    plt.close(fig)
    return cluster_to_id, cluster_to_color


def main() -> None:
    if not SELECTION.exists():
        raise FileNotFoundError(f"Run Figure 2A first; missing {SELECTION}")
    selection = pd.read_csv(SELECTION)
    representative_samples = selection.loc[
        selection["selected"].astype(bool), "sample_id"
    ].astype(str).tolist()
    if len(representative_samples) != 1:
        raise ValueError("Figure 2A selection audit must contain one representative sample")
    representative_sample = representative_samples[0]

    adata = ad.read_h5ad(INPUT)
    early_mask = adata.obs["stage"].astype(str).isin(["Stage I", "Stage II"]).to_numpy()
    obs = adata.obs.loc[
        early_mask, ["sample_id", "patient_id", "stage"]
    ].copy()
    obs["sample_id"] = obs["sample_id"].astype(str)
    obs["patient_id"] = obs["patient_id"].astype(str)
    proportions = np.asarray(adata.obsm["cell2location_proportions"])[early_mask]
    coordinates = np.asarray(adata.obsm["spatial"])[early_mask]
    sample_ids = obs["sample_id"].to_numpy()
    patient_ids = obs["patient_id"].to_numpy()

    local, spatial_graphs = build_local_profiles(proportions, coordinates, sample_ids)
    features = patient_centered_clr(local, patient_ids)
    train_index = balanced_training_index(patient_ids)
    k_audit, fitted = evaluate_k_values(
        features, train_index, patient_ids, sample_ids, spatial_graphs
    )
    selected_k = int(k_audit.loc[k_audit["selected"], "n_niches"].iloc[0])
    _, labels = fitted[selected_k]

    matrix, enrichment, by_patient = enrichment_tables(
        local, labels, patient_ids, selected_k
    )
    names = infer_niche_names(matrix)
    cluster_to_id, cluster_to_color = plot_figure(
        adata,
        early_mask,
        obs,
        coordinates,
        labels,
        matrix,
        enrichment,
        names,
        representative_sample,
    )

    enrichment["niche_id"] = enrichment["cluster"].map(cluster_to_id)
    enrichment["niche_name"] = enrichment["cluster"].map(names)
    enrichment["niche_color"] = enrichment["cluster"].map(cluster_to_color)
    by_patient["niche_id"] = by_patient["cluster"].map(cluster_to_id)
    by_patient["niche_name"] = by_patient["cluster"].map(names)

    assignment = obs.reset_index(names="spot_id")
    assignment["x"] = coordinates[:, 0]
    assignment["y"] = coordinates[:, 1]
    assignment["cluster"] = labels
    assignment["niche_id"] = assignment["cluster"].map(cluster_to_id)
    assignment["niche_name"] = assignment["cluster"].map(names)
    assignment["niche_color"] = assignment["cluster"].map(cluster_to_color)

    prevalence = (
        assignment.groupby(
            ["patient_id", "sample_id", "stage", "niche_id", "niche_name"],
            observed=True,
        )
        .size()
        .rename("n_spots")
        .reset_index()
    )
    prevalence["fraction_within_sample"] = prevalence["n_spots"] / prevalence.groupby(
        "sample_id", observed=True
    )["n_spots"].transform("sum")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    k_audit.to_csv(K_AUDIT, index=False)
    assignment.to_csv(ASSIGNMENT_AUDIT, index=False, compression="gzip")
    enrichment.sort_values(["niche_id", "cell_type"]).to_csv(
        ENRICHMENT_AUDIT, index=False
    )
    by_patient.sort_values(["patient_id", "niche_id", "cell_type"]).to_csv(
        PATIENT_ENRICHMENT_AUDIT, index=False
    )
    prevalence.sort_values(["patient_id", "niche_id"]).to_csv(
        PREVALENCE_AUDIT, index=False
    )
    metadata = {
        "input": str(INPUT.relative_to(ROOT)),
        "group": "Early-stage OSCC (Stage I-II)",
        "samples": sorted(set(sample_ids)),
        "patients": sorted(set(patient_ids)),
        "n_spots": int(len(obs)),
        "representative_sample": representative_sample,
        "input_view": "cell2location_proportions",
        "local_profile": "center spot plus five nearest spots, arithmetic mean",
        "transform": "CLR followed by within-patient median centering",
        "training": f"up to {TRAIN_PER_PATIENT} spots per patient with equal patient weight",
        "seed": SEED,
        "candidate_k": list(K_CANDIDATES),
        "selection_rule": {
            "strategy": "largest eligible K",
            "bootstrap_ari_mean_min": MIN_BOOTSTRAP_ARI,
            "minimum_group_fraction": MIN_GROUP_FRACTION,
            "minimum_patient_fraction_for_presence": MIN_PATIENT_FRACTION,
            "minimum_patients_per_niche": MIN_PATIENTS_PRESENT,
        },
        "selected_k": selected_k,
        "niche_names": {
            cluster_to_id[cluster]: names[cluster] for cluster in sorted(names)
        },
        "heatmap": "patient-level log2 observed/expected, group median",
        "dot_definition": (
            "same enrichment direction in >=3/4 patients and absolute median "
            f"log2(Ro/e) >= {DOT_MIN_ABS_LOG2_ROE}"
        ),
        "interpretation": (
            "Composition-defined spatial niches; functional immune labels require "
            "independent gene-program or ligand-receptor validation"
        ),
    }
    METADATA_AUDIT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"Selected K={selected_k}; {len(obs):,} spots from "
        f"{len(set(patient_ids))} Early-stage patients"
    )
    print(
        "Niches: "
        + ", ".join(
            f"{cluster_to_id[cluster]}={names[cluster]}" for cluster in sorted(names)
        )
    )
    print(f"Saved {OUTPUT.with_suffix('.png')}")
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")
    print(f"Saved audits under {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
