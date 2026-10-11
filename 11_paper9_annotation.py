"""Align the integrated single-cell atlas to the GSE181919/Visium nine classes.

Existing PCA, BBKNN, UMAP and Leiden results are reused unchanged. GSE181919
author labels are the reference truth. Labels from the other cohorts are only
mapped when their author definition is compatible with one of the nine classes;
dataset-specific populations outside the reference taxonomy remain in the
object but receive a missing paper9 label and are excluded from paper9 analyses.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "results/integrated_3ds_annotated_literature_v3.h5ad"
FALLBACK_SOURCE = ROOT / "data/processed/integrated_3ds.h5ad"
TARGET = ROOT / "results/integrated_3ds_annotated_paper9.h5ad"
RESULT = ROOT / "results"
FIGURE = ROOT / "figures/paper9"
PDF_FIGURE = FIGURE / "pdf"

REFERENCE = "Choi et al. 2023; doi:10.1038/s41467-023-36691-x"

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

# Marker panel used to audit the original GSE181919 major labels. These are
# identical to the author-label audit panel already used in literature-v3.
MARKERS = {
    "B/Plasma cells": ["CD79A", "MS4A1", "MZB1"],
    "Dendritic cells": ["CD1C", "FCER1A", "HLA-DRA"],
    "Endothelial cells": ["PECAM1", "VWF", "CLDN5"],
    "Epithelial cells": ["EPCAM", "KRT5", "KRT14", "KRT19"],
    "Fibroblasts": ["COL1A1", "DCN", "LUM"],
    "Macrophages": ["C1QA", "CD68", "LYZ"],
    "Mast cells": ["TPSAB1", "TPSB2", "CPA3"],
    "Myocytes": ["ACTA1", "ACTN2", "MYL2", "MYH2"],
    "NK/T cells": ["CD3D", "TRAC", "NKG7"],
}

DIRECT_AUTHOR_MAP = {
    "A4_GSE181919": {
        "B_Plasma.cells": "B/Plasma cells",
        "Dendritic.cells": "Dendritic cells",
        "Endothelial.cells": "Endothelial cells",
        "Epithelial.cells": "Epithelial cells",
        "Malignant.cells": "Epithelial cells",
        "Fibroblasts": "Fibroblasts",
        "Macrophages": "Macrophages",
        "Mast.cells": "Mast cells",
        "Myocytes": "Myocytes",
        "T.cells": "NK/T cells",
    },
    "B5_GSE310797": {
        "B cells": "B/Plasma cells",
        "CD4 T cells": "NK/T cells",
        "CD8 T cells": "NK/T cells",
        "Endothelial cells": "Endothelial cells",
        "Epithelial cells": "Epithelial cells",
        "Fibroblasts": "Fibroblasts",
        "Mast cells": "Mast cells",
    },
}

# GSE164241 has no GEO per-cell labels. These assignments reuse the existing
# paper-guided, within-dataset Leiden reconstruction without reclustering.
GSE164241_CLUSTER_MAP = {
    "0": "Endothelial cells",
    "1": "NK/T cells",
    "2": "Epithelial cells",
    "3": "Fibroblasts",
    "4": "NK/T cells",
    "5": "Macrophages",
    "7": "Fibroblasts",
    "8": "B/Plasma cells",
    "9": "B/Plasma cells",
    "11": "Epithelial cells",
    "12": "Mast cells",
    "14": "Endothelial cells",
    "16": "Dendritic cells",
}

EXCLUDED_AUTHOR_LABELS = {
    "B5_GSE310797": {
        "Pericytes": "Pericytes are not equivalent to GSE181919 skeletal myocytes",
        "Nerve-like cells": "Author-defined CNTNAP2 population is outside the paper9 taxonomy",
    }
}

CELL_COLORS = {
    "B/Plasma cells": "#4E9F8A",
    "Dendritic cells": "#8A6BB8",
    "Endothelial cells": "#4AA6A8",
    "Epithelial cells": "#D95F59",
    "Fibroblasts": "#6A9F58",
    "Macrophages": "#C28E2C",
    "Mast cells": "#C05A87",
    "Myocytes": "#B06B3C",
    "NK/T cells": "#3B78B5",
}


def decode(values) -> np.ndarray:
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)


def read_categorical(group: h5py.Group, name: str) -> np.ndarray:
    item = group[name]
    categories = decode(item["categories"][:])
    codes = item["codes"][:]
    output = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    output[valid] = categories[codes[valid]]
    return output


def extract_expression(h5: h5py.File, requested: set[str], chunk_size: int = 5000):
    genes = decode(h5["var"]["_index"][:])
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    present = [gene for gene in genes if gene in requested]
    selected = np.asarray([gene_to_index[gene] for gene in present], dtype=np.int32)
    n_cells, n_genes = (int(value) for value in h5["X"].attrs["shape"])
    output = np.zeros((n_cells, len(selected)), dtype=np.float32)
    indptr = h5["X"]["indptr"][:]
    for start in range(0, n_cells, chunk_size):
        end = min(start + chunk_size, n_cells)
        data_start, data_end = int(indptr[start]), int(indptr[end])
        chunk = sparse.csr_matrix(
            (
                h5["X"]["data"][data_start:data_end],
                h5["X"]["indices"][data_start:data_end],
                indptr[start:end + 1] - data_start,
            ),
            shape=(end - start, n_genes),
        )
        output[start:end] = chunk[:, selected].toarray()
    return output, np.asarray(present, dtype=object)


def score_program(expression: np.ndarray, genes: np.ndarray, marker_genes: list[str]) -> np.ndarray:
    gene_to_col = {gene: index for index, gene in enumerate(genes)}
    columns = [gene_to_col[gene] for gene in marker_genes if gene in gene_to_col]
    if not columns:
        return np.zeros(expression.shape[0], dtype=np.float32)
    return expression[:, columns].mean(axis=1)


def annotate(dataset, original, leiden, expression, genes):
    n_cells = len(dataset)
    labels = np.full(n_cells, "", dtype=object)
    basis = np.full(n_cells, "excluded from paper9", dtype=object)
    note = np.full(n_cells, "No defensible mapping to a GSE181919 paper9 class", dtype=object)

    for dataset_name, mapping in DIRECT_AUTHOR_MAP.items():
        for author_label, paper9_label in mapping.items():
            mask = (dataset == dataset_name) & (original == author_label)
            labels[mask] = paper9_label
            basis[mask] = "author label mapped to GSE181919 paper9 class"
            note[mask] = ""

    c6 = dataset == "C6_GSE164241"
    for cluster, paper9_label in GSE164241_CLUSTER_MAP.items():
        mask = c6 & (leiden == cluster)
        labels[mask] = paper9_label
        basis[mask] = "Williams2021 paper-marker reconstruction; existing Leiden reused"
        note[mask] = ""

    # Preserve the GSE164241 H.Mel population as excluded even when melanocytes
    # are embedded in a broader Leiden cluster.
    mel_score = sum(
        expression[:, np.flatnonzero(genes == gene)[0]] > 0
        for gene in ["DCT", "PMEL", "TYRP1", "MITF"] if np.any(genes == gene)
    )
    melanocyte = c6 & (mel_score >= 2)
    labels[melanocyte] = ""
    basis[melanocyte] = "excluded from paper9"
    note[melanocyte] = "GSE164241 H.Mel population is outside the paper9 taxonomy"

    for dataset_name, exclusions in EXCLUDED_AUTHOR_LABELS.items():
        for author_label, reason in exclusions.items():
            mask = (dataset == dataset_name) & (original == author_label)
            labels[mask] = ""
            basis[mask] = "excluded from paper9"
            note[mask] = reason

    # GSE310797 supplies one coarse Myeloid label. Split it without changing
    # clusters by comparing the original macrophage and dendritic marker panels.
    coarse = (dataset == "B5_GSE310797") & (original == "Myeloid cells")
    macrophage_score = score_program(expression, genes, MARKERS["Macrophages"])
    dendritic_score = score_program(expression, genes, MARKERS["Dendritic cells"])
    dc_positive = sum(
        expression[:, np.flatnonzero(genes == gene)[0]] > 0
        for gene in MARKERS["Dendritic cells"] if np.any(genes == gene)
    )
    assign_dc = coarse & (dc_positive > 0) & (dendritic_score > macrophage_score)
    assign_macro = coarse & ~assign_dc
    labels[assign_dc] = "Dendritic cells"
    labels[assign_macro] = "Macrophages"
    basis[coarse] = "GSE310797 coarse Myeloid split by GSE181919 original markers"
    note[assign_dc] = "DC marker score exceeded macrophage marker score"
    note[assign_macro] = "Macrophage marker score was greater or equal; default for coarse Myeloid"

    return labels, basis, note, macrophage_score, dendritic_score


def marker_statistics(obs: pd.DataFrame, expression: np.ndarray, genes: np.ndarray) -> pd.DataFrame:
    gene_to_col = {gene: index for index, gene in enumerate(genes)}
    rows = []
    scopes = [("ALL", np.ones(len(obs), dtype=bool))]
    scopes.extend((name, obs["dataset"].to_numpy() == name) for name in sorted(obs["dataset"].unique()))
    for scope, scope_mask in scopes:
        for cell_type in CELL_TYPES:
            mask = scope_mask & (obs["cell_type_paper9"].to_numpy() == cell_type)
            for expected_type, marker_genes in MARKERS.items():
                for gene in marker_genes:
                    if gene not in gene_to_col:
                        continue
                    values = expression[mask, gene_to_col[gene]]
                    rows.append({
                        "dataset": scope,
                        "cell_type": cell_type,
                        "marker_expected_for": expected_type,
                        "gene": gene,
                        "mean_expression": float(values.mean()) if len(values) else 0.0,
                        "pct_expressed": float((values > 0).mean() * 100) if len(values) else 0.0,
                        "n_cells": int(mask.sum()),
                    })
    return pd.DataFrame(rows)


def save_audits(obs, marker_stats, macrophage_score, dendritic_score):
    obs.to_csv(
        RESULT / "paper9_cell_annotation.csv.gz", index=False,
        encoding="utf-8-sig", compression="gzip",
    )
    counts = pd.crosstab(obs["dataset"], obs["cell_type_paper9"]).reindex(
        columns=CELL_TYPES, fill_value=0
    )
    counts["paper9_excluded"] = obs.groupby("dataset")["paper9_in_scope"].apply(
        lambda values: int((values == "excluded").sum())
    )
    counts.to_csv(RESULT / "paper9_dataset_counts.csv", encoding="utf-8-sig")
    pd.crosstab(
        [obs["dataset"], obs["original_celltype"]], obs["cell_type_paper9"], dropna=False
    ).reindex(columns=CELL_TYPES, fill_value=0).to_csv(
        RESULT / "paper9_author_mapping_counts.csv", encoding="utf-8-sig"
    )
    obs[[
        "dataset", "original_celltype", "leiden", "cell_type_paper9",
        "paper9_in_scope", "paper9_annotation_basis", "paper9_annotation_note",
    ]].drop_duplicates().sort_values([
        "dataset", "paper9_in_scope", "cell_type_paper9", "original_celltype", "leiden"
    ]).to_csv(RESULT / "paper9_mapping_audit.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame([
        {
            "cell_type": cell_type,
            "marker_genes": ";".join(markers),
            "reference": REFERENCE,
            "role": "GSE181919 paper9 annotation audit",
        }
        for cell_type, markers in MARKERS.items()
    ]).to_csv(RESULT / "paper9_marker_panel.csv", index=False, encoding="utf-8-sig")
    marker_stats.to_csv(RESULT / "paper9_marker_stats.csv", index=False, encoding="utf-8-sig")

    coarse = (obs["dataset"] == "B5_GSE310797") & (obs["original_celltype"] == "Myeloid cells")
    split = obs.loc[coarse, ["cell_id", "patient_id", "cell_type_paper9"]].copy()
    split["macrophage_marker_score"] = macrophage_score[coarse]
    split["dendritic_marker_score"] = dendritic_score[coarse]
    split.to_csv(
        RESULT / "paper9_GSE310797_myeloid_split.csv.gz", index=False,
        encoding="utf-8-sig", compression="gzip",
    )


def plot_marker_audit(marker_stats: pd.DataFrame):
    display_genes = [gene for cell_type in CELL_TYPES for gene in MARKERS[cell_type]]
    stats = marker_stats[
        (marker_stats["dataset"] == "ALL")
        & marker_stats["cell_type"].isin(CELL_TYPES)
        & marker_stats["gene"].isin(display_genes)
    ]
    means = stats.pivot(index="gene", columns="cell_type", values="mean_expression").reindex(
        index=display_genes, columns=CELL_TYPES
    )
    pct = stats.pivot(index="gene", columns="cell_type", values="pct_expressed").reindex(
        index=display_genes, columns=CELL_TYPES
    )
    scaled = means.copy()
    for gene in display_genes:
        values = means.loc[gene].to_numpy(float)
        scaled.loc[gene] = (values - values.mean()) / (values.std() + 1e-8)

    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    for y, gene in enumerate(display_genes):
        for x, cell_type in enumerate(CELL_TYPES):
            ax.scatter(
                x, y, s=12 + 1.5 * pct.loc[gene, cell_type],
                c=scaled.loc[gene, cell_type], cmap="RdBu_r", vmin=-1.5, vmax=2.5,
                edgecolors="none",
            )
    ax.set_xticks(range(len(CELL_TYPES)), [name.replace(" cells", "") for name in CELL_TYPES], rotation=40, ha="right")
    ax.set_yticks(range(len(display_genes)), display_genes)
    ax.invert_yaxis()
    ax.set_title("GSE181919 paper9 annotation: original-marker audit")
    ax.set_xlabel("Paper9 cell type")
    ax.set_ylabel("Marker gene")
    ax.grid(color="#E6E6E6", linewidth=0.5)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        output_dir = PDF_FIGURE if extension == "pdf" else FIGURE
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"Fig2E_paper9_annotation_markers.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_categorical(group, name, values, categories=None):
    if name in group:
        del group[name]
    values = np.asarray(values, dtype=object)
    if categories is None:
        categories = list(dict.fromkeys(value for value in values.tolist() if value != ""))
    category_to_code = {value: index for index, value in enumerate(categories)}
    codes = np.asarray([category_to_code.get(value, -1) for value in values], dtype=np.int16)
    item = group.create_group(name)
    item.attrs["encoding-type"] = "categorical"
    item.attrs["encoding-version"] = "0.2.0"
    item.attrs["ordered"] = False
    item.create_dataset("categories", data=np.asarray(categories, dtype=h5py.string_dtype("utf-8")))
    item.create_dataset("codes", data=codes)


def write_annotated_h5ad(obs: pd.DataFrame):
    if not TARGET.exists():
        source = SOURCE if SOURCE.exists() else FALLBACK_SOURCE
        print(f"Copying {source.name} to {TARGET.name}...", flush=True)
        shutil.copy2(source, TARGET)
    with h5py.File(TARGET, "r+") as h5:
        group = h5["obs"]
        current = read_categorical(group, "cell_type")
        if "cell_type_pre_paper9" not in group:
            write_categorical(group, "cell_type_pre_paper9", current)
        write_categorical(group, "cell_type", obs["cell_type_paper9"], categories=CELL_TYPES)
        write_categorical(group, "cell_type_paper9", obs["cell_type_paper9"], categories=CELL_TYPES)
        write_categorical(group, "paper9_in_scope", obs["paper9_in_scope"], categories=["included", "excluded"])
        write_categorical(group, "paper9_annotation_basis", obs["paper9_annotation_basis"])
        write_categorical(group, "paper9_annotation_note", obs["paper9_annotation_note"])
        columns = decode(group.attrs["column-order"]).tolist()
        additions = [
            "cell_type_pre_paper9", "cell_type_paper9", "paper9_in_scope",
            "paper9_annotation_basis", "paper9_annotation_note",
        ]
        del group.attrs["column-order"]
        group.attrs["column-order"] = np.asarray(
            list(dict.fromkeys(columns + additions)), dtype=h5py.string_dtype("utf-8")
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-h5ad", action="store_true")
    args = parser.parse_args()
    read_path = SOURCE if SOURCE.exists() else FALLBACK_SOURCE
    requested = {gene for markers in MARKERS.values() for gene in markers}
    requested.update(["DCT", "PMEL", "TYRP1", "MITF"])
    with h5py.File(read_path, "r") as h5:
        group = h5["obs"]
        cell_id = decode(group["_index"][:])
        dataset = read_categorical(group, "dataset")
        patient = read_categorical(group, "patient_id")
        disease = read_categorical(group, "disease_status")
        leiden = read_categorical(group, "leiden")
        original = read_categorical(group, "original_celltype")
        expression, genes = extract_expression(h5, requested)

    labels, basis, note, macrophage_score, dendritic_score = annotate(
        dataset, original, leiden, expression, genes
    )
    in_scope = np.where(labels == "", "excluded", "included")
    obs = pd.DataFrame({
        "cell_id": cell_id,
        "dataset": dataset,
        "patient_id": patient,
        "disease_status": disease,
        "leiden": leiden,
        "original_celltype": original,
        "cell_type_paper9": labels,
        "paper9_in_scope": in_scope,
        "paper9_annotation_basis": basis,
        "paper9_annotation_note": note,
    })
    marker_stats = marker_statistics(obs, expression, genes)
    save_audits(obs, marker_stats, macrophage_score, dendritic_score)
    plot_marker_audit(marker_stats)
    if args.write_h5ad:
        write_annotated_h5ad(obs)

    counts = pd.crosstab(obs["dataset"], obs["cell_type_paper9"]).reindex(columns=CELL_TYPES, fill_value=0)
    print(counts.to_string(), flush=True)
    print("Excluded:", obs.loc[obs["paper9_in_scope"] == "excluded", "dataset"].value_counts().to_dict(), flush=True)
    print("paper9 annotation complete", flush=True)


if __name__ == "__main__":
    main()
