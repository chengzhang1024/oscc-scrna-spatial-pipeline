"""Regenerate Figure 2A-G from the literature-first v3 annotation.

UMAP coordinates are reused from the integrated object. Cell-composition Ro/e
uses sample-balanced proportions so samples with more captured cells do not
dominate a disease group.
"""

from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
H5AD = ROOT / "results/integrated_3ds_annotated_literature_v3.h5ad"
RESULT = ROOT / "results"
FIGURE = ROOT / "figures"
PDF_FIGURE = FIGURE / "pdf"

CELL_TYPES = [
    "Epithelial cells", "T/NK cells", "B/Plasma cells", "Myeloid cells",
    "Fibroblasts", "Endothelial cells", "Perivascular cells", "Mast cells",
    "Myocytes", "Melanocytes", "Nerve-like cells (author-defined)",
    "Other/Uncertain",
]
COMMON_TYPES = [
    "Epithelial cells", "T/NK cells", "B/Plasma cells", "Myeloid cells",
    "Fibroblasts", "Endothelial cells", "Perivascular cells", "Mast cells",
]

DATASET_COLORS = {
    "A4_GSE181919": "#3B78B5",
    "B5_GSE310797": "#D95F59",
    "C6_GSE164241": "#4E9F8A",
}
DISEASE_COLORS = {
    "Normal": "#3B78B5", "OLK": "#C28E2C", "OSCC": "#D95F59",
    "Lymph_Node": "#8A6BB8",
}
CELL_COLORS = {
    "Epithelial cells": "#D95F59",
    "T/NK cells": "#3B78B5",
    "B/Plasma cells": "#4E9F8A",
    "Myeloid cells": "#C28E2C",
    "Fibroblasts": "#6A9F58",
    "Endothelial cells": "#4AA6A8",
    "Perivascular cells": "#8A6BB8",
    "Mast cells": "#C05A87",
    "Myocytes": "#B06B3C",
    "Melanocytes": "#262626",
    "Nerve-like cells (author-defined)": "#9B8065",
    "Other/Uncertain": "#BDBDBD",
}
STAGE_COLORS = {"Early-stage": "#3B78B5", "Late-stage": "#D95F59"}

STAGE_MAP = {
    "GSM9309214": ("PT1", "I", "Early-stage"),
    "GSM9309215": ("PT2", "II", "Early-stage"),
    "GSM9309216": ("PT3", "II", "Early-stage"),
    "GSM9309217": ("PT4", "II", "Early-stage"),
    "GSM9309218": ("PT5", "II", "Early-stage"),
    "GSM9309219": ("PT6", "II", "Early-stage"),
    "GSM9309220": ("PT7", "I", "Early-stage"),
    "GSM9309221": ("PT8", "III", "Late-stage"),
    "GSM9309222": ("PT9", "III", "Late-stage"),
    "GSM9309223": ("PT10", "III", "Late-stage"),
    "GSM9309224": ("PT11", "IV", "Late-stage"),
    "GSM9309225": ("PT12", "III", "Late-stage"),
    "GSM9309226": ("PT13", "III", "Late-stage"),
    "GSM9309227": ("PT14", "III", "Late-stage"),
    "GSM9309228": ("PT15", "III", "Late-stage"),
    "GSM9309229": ("PT16", "III", "Late-stage"),
}


def decode(values):
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)


def read_categorical(group, name):
    item = group[name]
    categories = decode(item["categories"][:])
    codes = item["codes"][:]
    output = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    output[valid] = categories[codes[valid]]
    return output


def balanced_subset(labels, max_per_group, seed=42):
    rng = np.random.default_rng(seed)
    indices = []
    for label in np.unique(labels):
        group = np.flatnonzero(labels == label)
        if len(group) > max_per_group:
            group = rng.choice(group, max_per_group, replace=False)
        indices.extend(group.tolist())
    return np.asarray(sorted(indices), dtype=np.int64)


def scatter(ax, coordinates, labels, order, colors, title, max_per_group):
    selected = balanced_subset(labels, max_per_group=max_per_group)
    for label in order:
        mask = labels[selected] == label
        if not mask.any():
            continue
        ax.scatter(
            coordinates[selected[mask], 0], coordinates[selected[mask], 1],
            s=1.0, alpha=0.58, linewidths=0, rasterized=True,
            color=colors[label], label=label,
        )
    ax.set_title(title, fontsize=10, weight="bold")
    ax.set_xlabel("UMAP1", fontsize=8)
    ax.set_ylabel("UMAP2", fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines[["top", "right", "bottom", "left"]].set_visible(False)
    if len(order) > 8:
        ax.legend(
            frameon=False, fontsize=6.5, markerscale=5,
            bbox_to_anchor=(1.01, 1), loc="upper left", borderaxespad=0,
        )
    else:
        ax.legend(frameon=False, fontsize=7, markerscale=5, loc="best")


def make_abcd(umap_pre, umap_post, dataset, disease, cell_type):
    fig, axes = plt.subplots(1, 4, figsize=(22.5, 5.0))
    scatter(
        axes[0], umap_pre, dataset, list(DATASET_COLORS), DATASET_COLORS,
        "A  Before batch correction", 30000,
    )
    scatter(
        axes[1], umap_post, dataset, list(DATASET_COLORS), DATASET_COLORS,
        "B  After BBKNN correction", 30000,
    )
    scatter(
        axes[2], umap_post, disease, ["Normal", "OLK", "OSCC", "Lymph_Node"],
        DISEASE_COLORS, "C  Disease status", 30000,
    )
    scatter(
        axes[3], umap_post, cell_type, CELL_TYPES, CELL_COLORS,
        "D  Literature-first harmonized types", 12000,
    )
    fig.suptitle("Integrated oral disease single-cell atlas (232,915 cells)", fontsize=13, weight="bold")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        output_dir = PDF_FIGURE if extension == "pdf" else FIGURE
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"Fig2ABCD_final_unified.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def sample_composition(obs, group_column):
    counts = pd.crosstab(
        [obs["dataset"], obs["patient_id"], obs[group_column]], obs["cell_type"]
    ).reindex(columns=CELL_TYPES, fill_value=0)
    proportions = counts.div(counts.sum(axis=1), axis=0)
    proportions.columns.name = "cell_type"
    return proportions


def sample_balanced_roe(proportions, group_level):
    groups = proportions.index.get_level_values(group_level)
    group_mean = proportions.groupby(groups, observed=True).mean()
    expected = proportions.mean(axis=0)
    return group_mean.div(expected.replace(0, np.nan), axis=1)


def plot_roe(roe, group_order, colors, title, output_name):
    cell_types = list(roe.columns)
    roe = roe.reindex(index=group_order, columns=cell_types)
    fig, ax = plt.subplots(figsize=(11.2, 4.7))
    x = np.arange(len(cell_types))
    for group in group_order:
        ax.plot(
            x, roe.loc[group], marker="o", linestyle="--", linewidth=1.8,
            markersize=6, color=colors[group], label=group,
            markeredgecolor="white", markeredgewidth=0.7,
        )
    ax.axhline(1, color="#777777", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([name.replace(" ", "\n") for name in cell_types], fontsize=8)
    ax.set_ylabel("Sample-balanced Ro/e")
    ax.set_title(title, fontsize=11, weight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, title="Group", bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        output_dir = PDF_FIGURE if extension == "pdf" else FIGURE
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{output_name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"Reading {H5AD.name}...", flush=True)
    with h5py.File(H5AD, "r") as h5:
        obs_group = h5["obs"]
        dataset = read_categorical(obs_group, "dataset")
        disease = read_categorical(obs_group, "disease_status")
        patient = read_categorical(obs_group, "patient_id")
        cell_type = read_categorical(obs_group, "cell_type")
        umap_pre = h5["obsm"]["X_umap_pre"][:]
        umap_post = h5["obsm"]["X_umap"][:]

    obs = pd.DataFrame({
        "dataset": dataset, "disease_status": disease,
        "patient_id": patient, "cell_type": cell_type,
    })
    make_abcd(umap_pre, umap_post, dataset, disease, cell_type)

    # Primary progression analysis: one cohort, one annotation, one processing workflow.
    primary = obs[
        (obs["dataset"] == "A4_GSE181919")
        & obs["disease_status"].isin(["Normal", "OLK", "OSCC"])
    ].copy()
    primary_prop = sample_composition(primary, "disease_status")
    primary_roe = sample_balanced_roe(primary_prop, "disease_status")
    primary_counts = pd.crosstab(
        [primary["dataset"], primary["patient_id"], primary["disease_status"]],
        primary["cell_type"],
    ).reindex(columns=CELL_TYPES, fill_value=0)
    sample_coverage = (primary_counts >= 10).groupby(level="disease_status").sum().reindex(
        ["Normal", "OLK", "OSCC"], fill_value=0
    )
    valid_primary_types = sample_coverage.columns[(sample_coverage >= 2).all(axis=0)].tolist()
    primary_roe = primary_roe[valid_primary_types]
    sample_coverage.to_csv(
        RESULT / "coverage_literature_v3_GSE181919_Normal_OLK_OSCC.csv", encoding="utf-8-sig"
    )
    plot_roe(
        primary_roe, ["Normal", "OLK", "OSCC"], DISEASE_COLORS,
        "F  Cell-type enrichment across progression (GSE181919)",
        "Fig2F_final_GSE181919_Normal_OLK_OSCC",
    )
    primary_roe.to_csv(RESULT / "roe_literature_v3_GSE181919_Normal_OLK_OSCC.csv", encoding="utf-8-sig")

    stage_rows = [
        {"patient_id": gsm, "patient": values[0], "stage": values[1], "stage_group": values[2]}
        for gsm, values in STAGE_MAP.items()
    ]
    pd.DataFrame(stage_rows).to_csv(
        RESULT / "literature_v3_GSE310797_confirmed_stage_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )
    staged = obs[obs["dataset"] == "B5_GSE310797"].copy()
    staged["oscc_stage"] = staged["patient_id"].map({gsm: value[2] for gsm, value in STAGE_MAP.items()})
    if staged["oscc_stage"].isna().any():
        raise ValueError("GSE310797 contains cells without a confirmed stage mapping")
    stage_prop = sample_composition(staged, "oscc_stage")
    stage_roe = sample_balanced_roe(stage_prop, "oscc_stage")
    stage_counts = pd.crosstab(
        [staged["dataset"], staged["patient_id"], staged["oscc_stage"]],
        staged["cell_type"],
    ).reindex(columns=CELL_TYPES, fill_value=0)
    stage_coverage = (stage_counts >= 10).groupby(level="oscc_stage").sum().reindex(
        ["Early-stage", "Late-stage"], fill_value=0
    )
    valid_stage_types = stage_coverage.columns[(stage_coverage >= 2).all(axis=0)].tolist()
    stage_roe = stage_roe[valid_stage_types]
    stage_coverage.to_csv(
        RESULT / "coverage_literature_v3_GSE310797_Early_Late.csv", encoding="utf-8-sig"
    )
    plot_roe(
        stage_roe, ["Early-stage", "Late-stage"], STAGE_COLORS,
        "G  Cell-type enrichment in early- versus late-stage OSCC",
        "Fig2G_final_GSE310797_Early_Late",
    )
    stage_roe.to_csv(RESULT / "roe_literature_v3_GSE310797_Early_Late.csv", encoding="utf-8-sig")

    primary_prop.reset_index().to_csv(
        RESULT / "composition_literature_v3_GSE181919_per_sample.csv", index=False, encoding="utf-8-sig"
    )
    stage_prop.reset_index().to_csv(
        RESULT / "composition_literature_v3_GSE310797_per_sample.csv", index=False, encoding="utf-8-sig"
    )
    print("Figure 2A-G literature-v3 complete.", flush=True)


if __name__ == "__main__":
    main()
