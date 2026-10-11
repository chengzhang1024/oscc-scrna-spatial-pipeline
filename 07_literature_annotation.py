"""Literature-first harmonization for GSE181919, GSE310797 and GSE164241.

Author-provided per-cell labels are retained for GSE181919 and GSE310797.
GSE164241 has no per-cell annotation in GEO, so its labels are reconstructed
within that dataset using the original paper's Table S4/S5 marker definitions.
The harmonization layer never lets an integrated Leiden cluster overwrite an
available author label.
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
SOURCE = ROOT / "data/processed/integrated_3ds.h5ad"
TARGET = ROOT / "results/integrated_3ds_annotated_literature_v3.h5ad"
RESULT = ROOT / "results"
FIGURE = ROOT / "figures"
PDF_FIGURE = FIGURE / "pdf"
REFERENCE = ROOT / "data/reference_annotation"

REFERENCES = {
    "A4_GSE181919": "Choi et al. 2023; doi:10.1038/s41467-023-36691-x",
    "B5_GSE310797": "Nat Commun 2026; doi:10.1038/s41467-026-70861-x",
    "C6_GSE164241": "Williams et al. 2021; doi:10.1016/j.cell.2021.05.013",
}

MAJOR_ORDER = [
    "Epithelial cells",
    "T/NK cells",
    "B/Plasma cells",
    "Myeloid cells",
    "Fibroblasts",
    "Endothelial cells",
    "Perivascular cells",
    "Myocytes",
    "Mast cells",
    "Melanocytes",
    "Nerve-like cells (author-defined)",
    "Other/Uncertain",
]

AUTHOR_MAP = {
    "A4_GSE181919": {
        "Epithelial.cells": "Epithelial cells",
        "Malignant.cells": "Epithelial cells",
        "T.cells": "T/NK cells",
        "B_Plasma.cells": "B/Plasma cells",
        "Macrophages": "Myeloid cells",
        "Dendritic.cells": "Myeloid cells",
        "Fibroblasts": "Fibroblasts",
        "Endothelial.cells": "Endothelial cells",
        "Myocytes": "Myocytes",
        "Mast.cells": "Mast cells",
    },
    "B5_GSE310797": {
        "Epithelial cells": "Epithelial cells",
        "CD4 T cells": "T/NK cells",
        "CD8 T cells": "T/NK cells",
        "B cells": "B/Plasma cells",
        "Myeloid cells": "Myeloid cells",
        "Fibroblasts": "Fibroblasts",
        "Endothelial cells": "Endothelial cells",
        "Pericytes": "Perivascular cells",
        "Mast cells": "Mast cells",
        # Retain the paper's CNTNAP2-defined label verbatim at the harmonized
        # layer, while marking it as dataset-specific and not cross-comparable.
        "Nerve-like cells": "Nerve-like cells (author-defined)",
    },
}

# Dataset-specific clusters are used only for GSE164241 because GEO contains no
# author per-cell labels. Assignments follow the original Table S4/S5 lineages.
GSE164241_CLUSTER_MAP = {
    "0": ("Endothelial cells", "H.VEC/H.LEC (paper-marker reconstruction)"),
    "1": ("T/NK cells", "T/NK (Table S5 reconstruction)"),
    "2": ("Epithelial cells", "H.Epi (Table S4 reconstruction)"),
    "3": ("Fibroblasts", "H.Fib (Table S4 reconstruction)"),
    "4": ("T/NK cells", "T/NK (Table S5 reconstruction)"),
    "5": ("Myeloid cells", "Granulocyte/Myeloid (Table S5 reconstruction)"),
    "6": ("Perivascular cells", "H.SMC (Table S4 reconstruction)"),
    "7": ("Fibroblasts", "H.Fib (Table S4 reconstruction)"),
    "8": ("B/Plasma cells", "B/Plasma (Table S5 reconstruction)"),
    "9": ("B/Plasma cells", "B/Plasma (Table S5 reconstruction)"),
    "11": ("Epithelial cells", "H.Epi (Table S4 reconstruction)"),
    "12": ("Mast cells", "Mast (original fine immune framework)"),
    "14": ("Endothelial cells", "H.VEC/H.LEC (paper-marker reconstruction)"),
    # HBA/HBB + epithelial + inflammatory/immunoglobulin signals indicate the
    # kind of contaminated/mixed cluster removed in the original workflow.
    "15": ("Other/Uncertain", "RBC/epithelial/immune mixed cluster"),
    # LILRA4/CLEC4C/GZMB/IRF7 identify a pDC-like myeloid population despite
    # JCHAIN/MZB1 expression.
    "16": ("Myeloid cells", "pDC-like myeloid (paper-guided reconstruction)"),
}

VALIDATION_MARKERS = {
    "Epithelial cells": ["KRT14", "KRT5", "EPCAM", "SFN"],
    "T/NK cells": ["CD3D", "TRAC", "NKG7", "GNLY"],
    "B/Plasma cells": ["CD79A", "MS4A1", "MZB1", "JCHAIN"],
    "Myeloid cells": ["LYZ", "TYROBP", "LST1", "C1QA"],
    "Fibroblasts": ["COL1A1", "DCN", "LUM", "COL3A1"],
    "Endothelial cells": ["PECAM1", "VWF", "CLDN5", "RAMP2"],
    "Perivascular cells": ["RGS5", "ACTA2", "MYH11", "TAGLN"],
    "Myocytes": ["ACTA1", "ACTN2", "MYL2", "MYH2"],
    "Mast cells": ["TPSAB1", "TPSB2", "CPA3", "KIT"],
    "Melanocytes": ["DCT", "PMEL", "TYRP1", "MITF"],
    "Nerve-like cells (author-defined)": ["CNTNAP2"],
}

AUTHOR_LABEL_AUDIT = {
    ("A4_GSE181919", "Epithelial.cells"): ["EPCAM", "KRT5", "KRT14"],
    ("A4_GSE181919", "Malignant.cells"): ["KRT19", "KRT13", "KRT18", "KRT8"],
    ("A4_GSE181919", "T.cells"): ["CD3D", "TRAC", "NKG7"],
    ("A4_GSE181919", "B_Plasma.cells"): ["CD79A", "MS4A1", "MZB1"],
    ("A4_GSE181919", "Macrophages"): ["C1QA", "CD68", "LYZ"],
    ("A4_GSE181919", "Dendritic.cells"): ["CD1C", "FCER1A", "HLA-DRA"],
    ("A4_GSE181919", "Fibroblasts"): ["COL1A1", "DCN", "LUM"],
    ("A4_GSE181919", "Endothelial.cells"): ["PECAM1", "VWF", "CLDN5"],
    ("A4_GSE181919", "Myocytes"): ["ACTA1", "ACTN2", "MYL2", "MYH2"],
    ("A4_GSE181919", "Mast.cells"): ["TPSAB1", "TPSB2", "CPA3"],
    ("B5_GSE310797", "Epithelial cells"): ["KRT5", "KRT14"],
    ("B5_GSE310797", "CD8 T cells"): ["CD3D", "CD8A"],
    ("B5_GSE310797", "CD4 T cells"): ["CD3D", "CD4"],
    ("B5_GSE310797", "B cells"): ["CD79A", "CD19"],
    ("B5_GSE310797", "Myeloid cells"): ["CD14", "LYZ"],
    ("B5_GSE310797", "Fibroblasts"): ["COL1A1", "TAGLN", "DCN"],
    ("B5_GSE310797", "Endothelial cells"): ["PECAM1", "VWF"],
    ("B5_GSE310797", "Pericytes"): ["RGS5", "TAGLN"],
    ("B5_GSE310797", "Mast cells"): ["KIT", "TPSAB1"],
    ("B5_GSE310797", "Nerve-like cells"): ["CNTNAP2"],
}


def decode(values):
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)


def read_categorical(group: h5py.Group, name: str) -> np.ndarray:
    item = group[name]
    categories = decode(item["categories"][:])
    codes = item["codes"][:]
    values = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    values[valid] = categories[codes[valid]]
    return values


def literature_signatures():
    s4_path = REFERENCE / "GSE164241_Table_S4_cluster_markers.xlsx"
    s5_path = REFERENCE / "GSE164241_Table_S5_immune_markers.xlsx"
    s4 = pd.read_excel(s4_path).sort_values(
        ["cluster", "avg_logFC"], ascending=[True, False]
    ).groupby("cluster", observed=True).head(20)
    s5 = pd.read_excel(s5_path).sort_values(
        ["cluster", "avg_logFC"], ascending=[True, False]
    ).groupby("cluster", observed=True).head(20)

    signatures = {
        "Epithelial cells": set(s4.loc[s4["cluster"].str.startswith("H.Epi"), "gene"]),
        "Fibroblasts": set(s4.loc[s4["cluster"].str.startswith("H.Fib"), "gene"]),
        "Endothelial cells": set(s4.loc[
            s4["cluster"].str.startswith("H.VEC") | s4["cluster"].eq("H.LEC"), "gene"
        ]),
        "Perivascular cells": set(s4.loc[s4["cluster"].eq("H.SMC"), "gene"]),
        "Melanocytes": set(s4.loc[s4["cluster"].eq("H.Mel"), "gene"]),
        "T/NK cells": set(s5.loc[s5["cluster"].eq("T/NK"), "gene"]),
        "B/Plasma cells": set(s5.loc[s5["cluster"].eq("B/Plasma"), "gene"]),
        "Myeloid cells": set(s5.loc[s5["cluster"].eq("Granulocyte/Myeloid"), "gene"]),
        "Mast cells": set(VALIDATION_MARKERS["Mast cells"] + ["MS4A2", "HDC"]),
    }
    rows = []
    for cell_type, genes in signatures.items():
        for gene in sorted(genes):
            rows.append({
                "harmonized_major_type": cell_type,
                "gene": gene,
                "source": REFERENCES["C6_GSE164241"],
                "use": "GSE164241 paper-guided reconstruction",
            })
    return signatures, pd.DataFrame(rows)


def extract_expression(h5: h5py.File, genes_requested, chunk_size=5000):
    genes = decode(h5["var"]["_index"][:])
    gene_to_index = {gene: index for index, gene in enumerate(genes)}
    present = sorted(set(genes_requested) & set(gene_to_index), key=gene_to_index.get)
    selected = np.asarray([gene_to_index[gene] for gene in present], dtype=np.int32)
    n_cells, n_genes = (int(x) for x in h5["X"].attrs["shape"])
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
        if start == 0 or end == n_cells or end % 50000 == 0:
            print(f"  expression extraction: {end:,}/{n_cells:,}", flush=True)
    return output, np.asarray(present, dtype=object)


def annotate(dataset, original, leiden, expression, genes):
    n_cells = len(dataset)
    major = np.full(n_cells, "Other/Uncertain", dtype=object)
    author_label = np.asarray(original, dtype=object).copy()
    basis = np.full(n_cells, "unresolved", dtype=object)
    reference = np.asarray([REFERENCES[item] for item in dataset], dtype=object)
    note = np.full(n_cells, "", dtype=object)

    for dataset_name, mapping in AUTHOR_MAP.items():
        dataset_mask = dataset == dataset_name
        for label, harmonized in mapping.items():
            mask = dataset_mask & (original == label)
            major[mask] = harmonized
            basis[mask] = "GEO author per-cell label -> harmonized major type"
        unknown = dataset_mask & (basis == "unresolved")
        note[unknown] = "Author label absent from literature mapping"

    c6 = dataset == "C6_GSE164241"
    for cluster, (harmonized, reconstructed) in GSE164241_CLUSTER_MAP.items():
        mask = c6 & (leiden == cluster)
        major[mask] = harmonized
        author_label[mask] = reconstructed
        basis[mask] = "Williams2021 Table S4/S5 marker reconstruction within GSE164241"
    mixed = c6 & (leiden == "15")
    basis[mixed] = "Williams2021-style contaminated/mixed cluster exclusion"
    note[mixed] = "RBC + epithelial + inflammatory/immunoglobulin signal; exclude from inference"
    pdc_like = c6 & (leiden == "16")
    note[pdc_like] = "LILRA4/CLEC4C/GZMB/IRF7 pDC-like program; retained in Myeloid"

    # Recover the H.Mel population defined in Table S4 even when it is embedded
    # in a broader integrated cluster. Requiring at least two markers limits
    # single-gene ambient RNA calls.
    gene_to_col = {gene: index for index, gene in enumerate(genes)}
    mel_cols = [gene_to_col[gene] for gene in ["DCT", "PMEL", "TYRP1", "MITF"] if gene in gene_to_col]
    mel_positive = np.zeros(n_cells, dtype=np.int16)
    if mel_cols:
        mel_positive = (expression[:, mel_cols] > 0).sum(axis=1)
    melanocyte = c6 & (mel_positive >= 2)
    major[melanocyte] = "Melanocytes"
    author_label[melanocyte] = "H.Mel (Table S4 marker reconstruction)"
    basis[melanocyte] = "Williams2021 Table S4 cell-level marker reconstruction"
    note[melanocyte] = "At least two of DCT/PMEL/TYRP1/MITF detected"

    author_nerve = (dataset == "B5_GSE310797") & (original == "Nerve-like cells")
    note[author_nerve] = (
        "Original CNTNAP2-defined label retained; dataset-specific and not "
        "independently asserted as Schwann/glial lineage"
    )
    return major, author_label, basis, reference, note, mel_positive


def signature_scores(expression, genes, c6_mask, signatures):
    gene_to_col = {gene: index for index, gene in enumerate(genes)}
    c6_expression = expression[c6_mask]
    means = c6_expression.mean(axis=0)
    stds = c6_expression.std(axis=0)
    z = np.divide(
        c6_expression - means, stds,
        out=np.zeros_like(c6_expression), where=stds > 0,
    )
    z = np.clip(z, -2, 5)
    scores = {}
    for cell_type, marker_set in signatures.items():
        columns = [gene_to_col[gene] for gene in marker_set if gene in gene_to_col]
        scores[cell_type] = z[:, columns].mean(axis=1) if columns else np.zeros(c6_mask.sum())
    return scores


def save_audits(obs, expression, genes, signatures, signature_reference):
    obs.to_csv(
        RESULT / "cell_annotation_literature_v3.csv.gz", index=False,
        encoding="utf-8-sig", compression="gzip",
    )
    pd.crosstab(obs["dataset"], obs["cell_type_literature_v3"]).reindex(
        columns=MAJOR_ORDER, fill_value=0
    ).to_csv(RESULT / "literature_v3_dataset_counts.csv", encoding="utf-8-sig")
    pd.crosstab(
        [obs["dataset"], obs["author_cell_type"]], obs["cell_type_literature_v3"]
    ).to_csv(RESULT / "literature_v3_author_mapping_counts.csv", encoding="utf-8-sig")
    obs[["dataset", "author_cell_type", "cell_type_literature_v3", "cross_dataset_status", "annotation_basis", "annotation_reference", "annotation_note"]].drop_duplicates().sort_values(
        ["dataset", "cell_type_literature_v3", "author_cell_type"]
    ).to_csv(RESULT / "literature_v3_mapping_reference.csv", index=False, encoding="utf-8-sig")
    signature_reference.to_csv(
        RESULT / "literature_v3_GSE164241_paper_signatures.csv", index=False, encoding="utf-8-sig"
    )

    c6_mask = obs["dataset"].to_numpy() == "C6_GSE164241"
    scores = signature_scores(expression, genes, c6_mask, signatures)
    c6_leiden = obs.loc[c6_mask, "leiden"].to_numpy()
    evidence_rows = []
    for cluster in sorted(set(c6_leiden), key=int):
        cluster_mask = c6_leiden == cluster
        assigned = GSE164241_CLUSTER_MAP.get(cluster, ("Other/Uncertain", "unmapped"))[0]
        row = {"leiden": cluster, "n_cells": int(cluster_mask.sum()), "assigned_major_type": assigned}
        for cell_type, values in scores.items():
            row[f"score_{cell_type}"] = float(values[cluster_mask].mean())
        evidence_rows.append(row)
    pd.DataFrame(evidence_rows).to_csv(
        RESULT / "literature_v3_GSE164241_cluster_evidence.csv", index=False, encoding="utf-8-sig"
    )

    marker_panel_rows = []
    for cell_type, marker_genes in VALIDATION_MARKERS.items():
        if cell_type == "Myocytes":
            reference = REFERENCES["A4_GSE181919"]
            scope = "GSE181919 dataset-specific author type"
        elif cell_type == "Melanocytes":
            reference = REFERENCES["C6_GSE164241"] + "; Table S4"
            scope = "GSE164241 dataset-specific reconstructed type"
        elif cell_type == "Nerve-like cells (author-defined)":
            reference = REFERENCES["B5_GSE310797"]
            scope = "GSE310797 dataset-specific author type"
        else:
            reference = "Original-paper labels/markers from all applicable datasets"
            scope = "Harmonized common major type"
        marker_panel_rows.append({
            "cell_type_literature_v3": cell_type,
            "marker_genes": ",".join(marker_genes),
            "marker_role": "Literature-consistency validation; not a replacement for author labels",
            "scope": scope,
            "reference": reference,
        })
    pd.DataFrame(marker_panel_rows).to_csv(
        RESULT / "celltype_markers_literature_v3.csv", index=False, encoding="utf-8-sig"
    )

    marker_rows = []
    gene_to_col = {gene: index for index, gene in enumerate(genes)}
    audit_groups = [("ALL", np.ones(len(obs), dtype=bool))]
    audit_groups += [(dataset, obs["dataset"].to_numpy() == dataset) for dataset in sorted(obs["dataset"].unique())]
    for dataset_name, dataset_mask in audit_groups:
        for cell_type in MAJOR_ORDER:
            mask = dataset_mask & (obs["cell_type_literature_v3"].to_numpy() == cell_type)
            for marker_type, marker_genes in VALIDATION_MARKERS.items():
                for gene in marker_genes:
                    if gene not in gene_to_col:
                        continue
                    values = expression[mask, gene_to_col[gene]]
                    marker_rows.append({
                        "dataset": dataset_name,
                        "cell_type": cell_type,
                        "marker_expected_for": marker_type,
                        "gene": gene,
                        "mean_expression": float(values.mean()) if len(values) else 0.0,
                        "pct_expressed": float((values > 0).mean() * 100) if len(values) else 0.0,
                        "n_cells": int(mask.sum()),
                    })
    marker_stats = pd.DataFrame(marker_rows)
    marker_stats.to_csv(
        RESULT / "literature_v3_marker_stats.csv", index=False, encoding="utf-8-sig"
    )

    author_audit_rows = []
    for (dataset_name, author_type), marker_genes in AUTHOR_LABEL_AUDIT.items():
        mask = (
            (obs["dataset"].to_numpy() == dataset_name)
            & (obs["author_cell_type"].to_numpy() == author_type)
        )
        for gene in marker_genes:
            if gene not in gene_to_col:
                continue
            values = expression[mask, gene_to_col[gene]]
            author_audit_rows.append({
                "dataset": dataset_name,
                "author_cell_type": author_type,
                "gene": gene,
                "mean_expression": float(values.mean()) if len(values) else 0.0,
                "pct_expressed": float((values > 0).mean() * 100) if len(values) else 0.0,
                "n_cells": int(mask.sum()),
                "reference": REFERENCES[dataset_name],
            })
    pd.DataFrame(author_audit_rows).to_csv(
        RESULT / "literature_v3_author_label_marker_audit.csv", index=False, encoding="utf-8-sig"
    )
    return marker_stats


def plot_marker_audit(marker_stats):
    display_markers = {
        "Epithelial cells": ["KRT14", "KRT5", "EPCAM"],
        "T/NK cells": ["CD3D", "TRAC", "NKG7"],
        "B/Plasma cells": ["CD79A", "MS4A1", "MZB1"],
        "Myeloid cells": ["LYZ", "TYROBP", "LST1"],
        "Fibroblasts": ["COL1A1", "DCN", "LUM"],
        "Endothelial cells": ["PECAM1", "VWF", "CLDN5"],
        "Perivascular cells": ["RGS5", "ACTA2", "MYH11"],
        "Myocytes": ["ACTA1", "ACTN2", "MYL2"],
        "Mast cells": ["TPSAB1", "TPSB2", "CPA3"],
        "Melanocytes": ["DCT", "PMEL", "TYRP1"],
        "Nerve-like cells (author-defined)": ["CNTNAP2"],
    }
    cell_types = list(display_markers)
    display_genes = [gene for cell_type in cell_types for gene in display_markers[cell_type]]
    stats = marker_stats[
        (marker_stats["dataset"] == "ALL")
        & marker_stats["cell_type"].isin(cell_types)
        & marker_stats["gene"].isin(display_genes)
    ].copy()
    means = stats.pivot(index="gene", columns="cell_type", values="mean_expression").reindex(
        index=display_genes, columns=cell_types
    )
    pct = stats.pivot(index="gene", columns="cell_type", values="pct_expressed").reindex(
        index=display_genes, columns=cell_types
    )
    scaled = means.copy()
    for gene in display_genes:
        values = means.loc[gene].to_numpy(float)
        scaled.loc[gene] = (values - values.mean()) / (values.std() + 1e-8)

    fig, ax = plt.subplots(figsize=(14.0, 9.2))
    for y, gene in enumerate(display_genes):
        for x, cell_type in enumerate(cell_types):
            ax.scatter(
                x, y,
                s=12 + 1.5 * pct.loc[gene, cell_type],
                c=scaled.loc[gene, cell_type], cmap="RdBu_r", vmin=-1.5, vmax=2.5,
                edgecolors="none",
            )
    ax.set_xticks(range(len(cell_types)), [item.replace(" cells", "") for item in cell_types], rotation=40, ha="right")
    ax.set_yticks(range(len(display_genes)), display_genes)
    ax.invert_yaxis()
    ax.set_title("Literature-first unified annotation: marker audit")
    ax.set_xlabel("Harmonized major type")
    ax.set_ylabel("Marker gene")
    ax.grid(color="#E6E6E6", linewidth=0.5)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        output_dir = PDF_FIGURE if extension == "pdf" else FIGURE
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            output_dir / f"Fig2E_final_annotation_markers.{extension}",
            dpi=300, bbox_inches="tight",
        )
    plt.close(fig)


def write_categorical(group, name, values):
    if name in group:
        del group[name]
    values = np.asarray(values, dtype=object)
    categories = list(dict.fromkeys(values.tolist()))
    category_to_code = {value: index for index, value in enumerate(categories)}
    item = group.create_group(name)
    item.attrs["encoding-type"] = "categorical"
    item.attrs["encoding-version"] = "0.2.0"
    item.attrs["ordered"] = False
    item.create_dataset("categories", data=np.asarray(categories, dtype=h5py.string_dtype("utf-8")))
    item.create_dataset("codes", data=np.asarray([category_to_code[value] for value in values], dtype=np.int16))


def write_annotated_h5ad(obs):
    if not TARGET.exists():
        print(f"Copying source object to {TARGET.name}...", flush=True)
        shutil.copy2(SOURCE, TARGET)
    elif TARGET.stat().st_size < SOURCE.stat().st_size * 0.95:
        raise RuntimeError(f"Existing target appears incomplete: {TARGET}")
    else:
        print(f"Reusing existing target {TARGET.name}...", flush=True)
    with h5py.File(TARGET, "r+") as h5:
        obs_group = h5["obs"]
        write_categorical(obs_group, "cell_type_pre_literature", obs["cell_type_pre_literature"])
        write_categorical(obs_group, "cell_type", obs["cell_type_literature_v3"])
        write_categorical(obs_group, "cell_type_literature_v3", obs["cell_type_literature_v3"])
        write_categorical(obs_group, "author_cell_type", obs["author_cell_type"])
        write_categorical(obs_group, "annotation_basis", obs["annotation_basis"])
        write_categorical(obs_group, "annotation_reference", obs["annotation_reference"])
        write_categorical(obs_group, "annotation_note", obs["annotation_note"])
        write_categorical(obs_group, "cross_dataset_status", obs["cross_dataset_status"])
        columns = decode(obs_group.attrs["column-order"]).tolist()
        additions = [
            "cell_type_pre_literature", "cell_type_literature_v3", "author_cell_type",
            "annotation_basis", "annotation_reference", "annotation_note", "cross_dataset_status",
        ]
        del obs_group.attrs["column-order"]
        obs_group.attrs["column-order"] = np.asarray(
            list(dict.fromkeys(columns + additions)), dtype=h5py.string_dtype("utf-8")
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-h5ad", action="store_true")
    args = parser.parse_args()
    signatures, signature_reference = literature_signatures()
    requested = set().union(*signatures.values())
    requested.update(gene for genes in VALIDATION_MARKERS.values() for gene in genes)
    requested.update(gene for genes in AUTHOR_LABEL_AUDIT.values() for gene in genes)

    with h5py.File(SOURCE, "r") as h5:
        obs_group = h5["obs"]
        cell_id = decode(obs_group["_index"][:])
        dataset = read_categorical(obs_group, "dataset")
        disease = read_categorical(obs_group, "disease_status")
        leiden = read_categorical(obs_group, "leiden")
        original = read_categorical(obs_group, "original_celltype")
        pre_literature = read_categorical(obs_group, "cell_type")
        expression, genes = extract_expression(h5, requested)

    major, author_label, basis, reference, note, mel_positive = annotate(
        dataset, original, leiden, expression, genes
    )
    common_major = {
        "Epithelial cells", "T/NK cells", "B/Plasma cells", "Myeloid cells",
        "Fibroblasts", "Endothelial cells", "Perivascular cells", "Mast cells",
    }
    cross_dataset_status = np.asarray([
        "common_major" if cell_type in common_major
        else "exclude_mixed" if cell_type == "Other/Uncertain"
        else "dataset_specific_author_type"
        for cell_type in major
    ], dtype=object)
    obs = pd.DataFrame({
        "cell_id": cell_id,
        "dataset": dataset,
        "disease_status": disease,
        "leiden": leiden,
        "original_celltype": original,
        "cell_type_pre_literature": pre_literature,
        "author_cell_type": author_label,
        "cell_type_literature_v3": major,
        "annotation_basis": basis,
        "annotation_reference": reference,
        "annotation_note": note,
        "cross_dataset_status": cross_dataset_status,
        "GSE164241_melanocyte_marker_count": mel_positive,
    })
    print("\nLiterature-first annotation counts:", flush=True)
    print(pd.crosstab(obs["dataset"], obs["cell_type_literature_v3"]).reindex(
        columns=MAJOR_ORDER, fill_value=0
    ).to_string(), flush=True)
    marker_stats = save_audits(obs, expression, genes, signatures, signature_reference)
    plot_marker_audit(marker_stats)
    if args.write_h5ad:
        write_annotated_h5ad(obs)
    print("Literature-v3 annotation complete.", flush=True)


if __name__ == "__main__":
    main()
