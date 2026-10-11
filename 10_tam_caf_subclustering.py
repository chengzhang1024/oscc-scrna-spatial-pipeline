"""Literature-guided Myeloid/Fibroblast subclustering and niche screening.

The analysis uses the literature-v3 integrated object, clusters each parent
compartment on batch-centered HVGs, annotates clusters with programs grounded
in the three source studies, and performs sample-level stage tests. Scanpy is
not required: the environment's Scanpy import is unstable, so the same core
steps are implemented with scipy, scikit-learn, and NetworkX Louvain.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import anndata as ad
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import kruskal, mannwhitneyu, spearmanr
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm


ROOT = Path(__file__).resolve().parent
H5AD = ROOT / "results/integrated_3ds_annotated_literature_v3.h5ad"
CELL_TYPE_FIELD = "cell_type_literature_v3"
RESULT = ROOT / "results"
FIGURE = ROOT / "figures"
PDF_FIGURE = FIGURE / "pdf"
SEED = 20260727
RNG = np.random.default_rng(SEED)

DATASET_LABELS = {
    "A4_GSE181919": "GSE181919",
    "B5_GSE310797": "GSE310797",
    "C6_GSE164241": "GSE164241",
}

MYELOID_PROGRAMS = {
    "MDSC-like": ["S100A8", "S100A9", "FCN1", "VCAN", "CTSS", "FPR1", "FPR2", "IL1B", "CXCL8"],
    "C1QC TAM": ["C1QA", "C1QB", "C1QC", "APOE", "CD163", "MRC1", "MSR1", "FOLR2", "FPR3"],
    "SPP1 TAM": ["SPP1", "APOC1", "TREM2", "GPNMB", "LPL", "CTSB", "CTSL"],
    "CD1C DC": ["CD1C", "FCER1A", "CLEC10A", "CD74", "HLA-DRA"],
    "pDC-like": ["GZMB", "JCHAIN", "CLEC4C", "LILRA4", "IRF7"],
    "Neutrophil-like": ["FCGR3B", "CSF3R", "CXCR2", "NAMPT", "S100A8", "S100A9"],
    "Cycling myeloid": ["MKI67", "TOP2A", "STMN1", "TUBA1B"],
}

FIBRO_PROGRAMS = {
    "CTHRC1/POSTN CAF": ["CTHRC1", "POSTN", "SPARC", "ASPN", "COL1A1", "COL3A1", "MFAP2", "MMP2"],
    "CXCL8 CAF": ["CXCL8", "CXCL2", "CXCL3", "IL6", "CCL2", "PTGS2", "NFKBIA"],
    "NDUFA4L2 myofibroblast": ["NDUFA4L2", "ACTA2", "TAGLN", "MYL9", "TPM2", "CALD1"],
    "CXCL12/CFD fibroblast": ["CXCL12", "CFD", "CXCL14", "DPT", "COL14A1", "APOD", "FBLN1"],
    "CCL19 immune fibroblast": ["CCL19", "CXCL13", "CCL2", "APOE", "HLA-DRA", "CD74"],
    "SPARCL1/PTGDS fibroblast": ["SPARCL1", "PTGDS", "CFH", "PLAC9", "SPRY1"],
    "APCDD1/IGFBP2 fibroblast": ["APCDD1", "IGFBP2", "DIO2", "LEPR", "SFRP2", "WNT5A"],
    "Cycling fibroblast": ["MKI67", "TOP2A", "STMN1", "TUBA1B"],
}

CONTAMINANT_PROGRAMS = {
    "Epithelial contaminant": ["KRT5", "KRT14", "KRT19", "EPCAM", "SFN"],
    "T/NK contaminant": ["CD3D", "TRAC", "NKG7", "GNLY"],
    "B/Plasma contaminant": ["CD79A", "MS4A1", "MZB1", "JCHAIN"],
    "Endothelial contaminant": ["PECAM1", "VWF", "CLDN5", "RAMP2"],
    "Mast contaminant": ["TPSAB1", "TPSB2", "CPA3", "KIT"],
}

CONFIGS = {
    "Myeloid": {
        "parent": "Myeloid cells",
        "programs": MYELOID_PROGRAMS,
        "resolution": 1.0,
        "candidate_labels": ["MDSC-like", "C1QC TAM", "SPP1 TAM"],
        "display_genes": [
            "S100A8", "S100A9", "FPR2", "C1QA", "C1QB", "CD163",
            "SPP1", "APOC1", "GPNMB", "CD1C", "FCER1A", "HLA-DRA",
            "GZMB", "CLEC4C", "LILRA4", "FCGR3B", "CSF3R", "CXCR2",
        ],
    },
    "Fibroblast": {
        "parent": "Fibroblasts",
        "programs": FIBRO_PROGRAMS,
        "resolution": 1.0,
        "candidate_labels": ["CTHRC1/POSTN CAF", "CXCL8 CAF", "NDUFA4L2 myofibroblast"],
        "display_genes": [
            "CTHRC1", "POSTN", "COL1A1", "CXCL8", "CXCL2", "IL6",
            "NDUFA4L2", "ACTA2", "TAGLN", "CXCL12", "CFD", "DPT",
            "CCL19", "CCL2", "CXCL13", "SPARCL1", "PTGDS", "CFH",
            "APCDD1", "IGFBP2", "SFRP2",
        ],
    },
}

PALETTE = [
    "#2A6FBB", "#D1495B", "#2A9D8F", "#E9A23B", "#7851A9", "#4D908E",
    "#E76F51", "#5C677D", "#8AB17D", "#C77DFF", "#577590", "#BC6C25",
]
GROUP_COLORS = {"Normal": "#3C8D70", "OLK": "#D9A441", "OSCC": "#C74B50"}
STAGE_COLORS = {"Early-stage": "#3F78B5", "Late-stage": "#C74B50"}


def excluded_gene(gene: str) -> bool:
    return (
        gene.startswith("MT-") or gene.startswith("RPL") or gene.startswith("RPS")
        or gene.startswith("IGH") or gene.startswith("IGK") or gene.startswith("IGL")
        or gene.startswith("TRAV") or gene.startswith("TRBV")
    )


def safe_spearman(x: np.ndarray | pd.Series, y: np.ndarray | pd.Series) -> tuple[float, float]:
    x_values = np.asarray(x, dtype=float)
    y_values = np.asarray(y, dtype=float)
    valid = np.isfinite(x_values) & np.isfinite(y_values)
    x_values = x_values[valid]
    y_values = y_values[valid]
    if len(x_values) < 3 or np.unique(x_values).size < 2 or np.unique(y_values).size < 2:
        return np.nan, np.nan
    result = spearmanr(x_values, y_values)
    return float(result.statistic), float(result.pvalue)


def mean_var(matrix: sparse.csr_matrix) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.asarray(matrix.mean(axis=0)).ravel()
    mean_sq = np.asarray(matrix.multiply(matrix).mean(axis=0)).ravel()
    var = np.maximum(mean_sq - mean * mean, 0)
    pct = np.asarray(matrix.getnnz(axis=0)).ravel() / matrix.shape[0]
    return mean, var, pct


def binned_dispersion(mean: np.ndarray, var: np.ndarray, valid: np.ndarray) -> np.ndarray:
    score = np.full(mean.shape, -np.inf, dtype=float)
    idx = np.flatnonzero(valid)
    if len(idx) == 0:
        return score
    log_mean = np.log1p(mean[idx])
    quantiles = np.unique(np.quantile(log_mean, np.linspace(0, 1, 21)))
    bins = np.digitize(log_mean, quantiles[1:-1], right=True)
    dispersion = np.log1p(var[idx] / (mean[idx] + 1e-4))
    normalized = np.zeros_like(dispersion)
    for bin_id in np.unique(bins):
        mask = bins == bin_id
        values = dispersion[mask]
        median = np.median(values)
        mad = np.median(np.abs(values - median)) * 1.4826
        normalized[mask] = (values - median) / (mad + 1e-6)
    score[idx] = normalized
    return score


def choose_hvgs(
    matrix: sparse.csr_matrix,
    genes: np.ndarray,
    datasets: np.ndarray,
    forced_genes: set[str],
    n_top: int = 1800,
) -> np.ndarray:
    allowed = np.asarray([not excluded_gene(gene) for gene in genes])
    rank_scores = []
    for dataset in sorted(set(datasets)):
        mask = datasets == dataset
        if mask.sum() < 100:
            continue
        mean, var, pct = mean_var(matrix[mask])
        valid = allowed & (pct >= 0.01) & (mean > 0.005)
        score = binned_dispersion(mean, var, valid)
        order = np.argsort(score)[::-1]
        rank = np.full(len(genes), 1.0)
        finite = order[np.isfinite(score[order])]
        rank[finite] = np.arange(len(finite)) / max(len(finite) - 1, 1)
        rank_scores.append(rank)
    combined = np.mean(rank_scores, axis=0)
    overall_pct = np.asarray(matrix.getnnz(axis=0)).ravel() / matrix.shape[0]
    combined[(~allowed) | (overall_pct < 0.005)] = np.inf
    selected = list(np.argsort(combined)[:n_top])
    gene_index = {gene: index for index, gene in enumerate(genes)}
    selected.extend(gene_index[gene] for gene in forced_genes if gene in gene_index)
    return np.asarray(sorted(set(selected)), dtype=int)


def batch_center_and_pca(
    matrix: sparse.csr_matrix,
    hvg_idx: np.ndarray,
    datasets: np.ndarray,
    n_components: int = 30,
) -> tuple[np.ndarray, np.ndarray]:
    selected = matrix[:, hvg_idx].astype(np.float32)
    selected = StandardScaler(with_mean=False).fit_transform(selected)
    n_components = min(n_components, selected.shape[0] - 1, selected.shape[1] - 1)
    pca = TruncatedSVD(
        n_components=n_components,
        algorithm="randomized",
        n_iter=7,
        random_state=SEED,
    )
    pcs = pca.fit_transform(selected).astype(np.float32)
    for dataset in sorted(set(datasets)):
        mask = datasets == dataset
        pcs[mask] -= pcs[mask].mean(axis=0, keepdims=True)
    pc_std = pcs.std(axis=0)
    pcs /= np.where(pc_std > 1e-4, pc_std, 1.0)
    return pcs, pca.explained_variance_ratio_


def graph_cluster(pcs: np.ndarray, resolution: float) -> tuple[np.ndarray, int]:
    n_neighbors = min(16, len(pcs))
    model = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean", n_jobs=4)
    distances, neighbors = model.fit(pcs).kneighbors(pcs)
    source = np.repeat(np.arange(len(pcs), dtype=np.int32), n_neighbors - 1)
    target = neighbors[:, 1:].reshape(-1).astype(np.int32)
    edge_distance = distances[:, 1:].reshape(-1)
    keep = source != target
    source, target, edge_distance = source[keep], target[keep], edge_distance[keep]
    scale = np.median(edge_distance[edge_distance > 0])
    weights = np.exp(-edge_distance / (scale + 1e-8)).astype(float)
    adjacency = sparse.csr_matrix(
        (weights, (source, target)), shape=(len(pcs), len(pcs))
    )
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.eliminate_zeros()
    graph = nx.from_scipy_sparse_array(adjacency, edge_attribute="weight")
    communities = nx.community.louvain_communities(
        graph, weight="weight", resolution=resolution, seed=SEED
    )
    labels = np.full(len(pcs), -1, dtype=int)
    for community_id, community in enumerate(communities):
        labels[np.fromiter(community, dtype=int)] = community_id
    counts = pd.Series(labels).value_counts()
    order = counts.sort_values(ascending=False).index.tolist()
    remap = {old: new for new, old in enumerate(order)}
    labels = np.asarray([remap[value] for value in labels], dtype=int)
    return labels, graph.number_of_edges()


def stratified_tsne(pcs: np.ndarray, clusters: np.ndarray, max_cells: int = 14000) -> tuple[np.ndarray, np.ndarray]:
    selected = []
    cluster_ids = sorted(set(clusters))
    base = max(250, max_cells // max(len(cluster_ids), 1))
    for cluster in cluster_ids:
        idx = np.flatnonzero(clusters == cluster)
        take = min(len(idx), base)
        selected.extend(RNG.choice(idx, take, replace=False).tolist())
    selected = np.asarray(sorted(set(selected)), dtype=int)
    if len(selected) > max_cells:
        selected = np.sort(RNG.choice(selected, max_cells, replace=False))
    perplexity = min(40, max(10, (len(selected) - 1) // 100))
    embedding = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate="auto",
        init="pca",
        max_iter=750,
        method="barnes_hut",
        angle=0.5,
        n_jobs=4,
        random_state=SEED,
    ).fit_transform(pcs[selected])
    return selected, embedding.astype(np.float32)


def cluster_expression(
    matrix: sparse.csr_matrix,
    clusters: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    cluster_ids = sorted(set(clusters))
    total_sum = np.asarray(matrix.sum(axis=0)).ravel()
    total_nnz = np.asarray(matrix.getnnz(axis=0)).ravel()
    means, pcts, marker_rows = [], [], []
    n_total = matrix.shape[0]
    for cluster in cluster_ids:
        mask = clusters == cluster
        n_in = int(mask.sum())
        sub = matrix[mask]
        sum_in = np.asarray(sub.sum(axis=0)).ravel()
        nnz_in = np.asarray(sub.getnnz(axis=0)).ravel()
        mean_in = sum_in / n_in
        pct_in = nnz_in / n_in
        n_out = max(n_total - n_in, 1)
        mean_out = (total_sum - sum_in) / n_out
        pct_out = (total_nnz - nnz_in) / n_out
        score = (mean_in - mean_out) * np.maximum(pct_in - pct_out, 0)
        means.append(mean_in)
        pcts.append(pct_in)
        marker_rows.append((cluster, n_in, score, mean_in, mean_out, pct_in, pct_out))
    return np.vstack(means), np.vstack(pcts), marker_rows


def program_scores_by_cluster(
    cluster_means: np.ndarray,
    genes: np.ndarray,
    programs: dict[str, list[str]],
) -> pd.DataFrame:
    gene_index = {gene: index for index, gene in enumerate(genes)}
    unique_genes = sorted({gene for values in programs.values() for gene in values if gene in gene_index})
    indices = [gene_index[gene] for gene in unique_genes]
    values = cluster_means[:, indices]
    z = (values - values.mean(axis=0, keepdims=True)) / (values.std(axis=0, keepdims=True) + 1e-6)
    z_by_gene = {gene: z[:, i] for i, gene in enumerate(unique_genes)}
    rows = []
    for cluster in range(cluster_means.shape[0]):
        row = {"subcluster": str(cluster)}
        for name, marker_genes in programs.items():
            present = [z_by_gene[gene][cluster] for gene in marker_genes if gene in z_by_gene]
            row[name] = float(np.mean(present)) if present else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def annotate_clusters(
    lineage_scores: pd.DataFrame,
    contaminant_scores: pd.DataFrame,
    programs: dict[str, list[str]],
) -> pd.DataFrame:
    rows = []
    program_names = list(programs)
    contaminant_names = list(CONTAMINANT_PROGRAMS)
    for _, row in lineage_scores.iterrows():
        cluster = str(row["subcluster"])
        lineage = row[program_names].astype(float).sort_values(ascending=False)
        contamination = contaminant_scores.loc[
            contaminant_scores["subcluster"] == cluster, contaminant_names
        ].iloc[0].astype(float).sort_values(ascending=False)
        top_name = lineage.index[0]
        top_score = float(lineage.iloc[0])
        second_score = float(lineage.iloc[1]) if len(lineage) > 1 else np.nan
        contaminant_name = contamination.index[0]
        contaminant_score = float(contamination.iloc[0])
        if contaminant_score > top_score + 0.35 and contaminant_score > 0.5:
            annotation = f"Excluded: {contaminant_name}"
        else:
            annotation = top_name
        rows.append({
            "subcluster": cluster,
            "subtype": annotation,
            "top_program_score": top_score,
            "second_program_score": second_score,
            "program_margin": top_score - second_score,
            "top_contaminant_program": contaminant_name,
            "top_contaminant_score": contaminant_score,
        })
    return pd.DataFrame(rows)


def marker_table(
    marker_rows: list,
    genes: np.ndarray,
    annotation: pd.DataFrame,
    top_n: int = 50,
) -> pd.DataFrame:
    label_map = annotation.set_index("subcluster")["subtype"].to_dict()
    rows = []
    allowed = np.asarray([not excluded_gene(gene) for gene in genes])
    for cluster, n_cells, score, mean_in, mean_out, pct_in, pct_out in marker_rows:
        order = np.argsort(np.where(allowed, score, -np.inf))[::-1][:top_n]
        for rank, index in enumerate(order, start=1):
            rows.append({
                "subcluster": str(cluster),
                "subtype": label_map[str(cluster)],
                "n_cells": n_cells,
                "rank": rank,
                "gene": genes[index],
                "marker_score": float(score[index]),
                "mean_in": float(mean_in[index]),
                "mean_out": float(mean_out[index]),
                "pct_in": float(pct_in[index] * 100),
                "pct_out": float(pct_out[index] * 100),
            })
    return pd.DataFrame(rows)


def per_cell_program_scores(matrix: sparse.csr_matrix, genes: np.ndarray, programs: dict[str, list[str]]) -> pd.DataFrame:
    gene_index = {gene: index for index, gene in enumerate(genes)}
    scores = {}
    for name, marker_genes in programs.items():
        idx = [gene_index[gene] for gene in marker_genes if gene in gene_index]
        scores[name] = np.asarray(matrix[:, idx].mean(axis=1)).ravel() if idx else np.nan
    return pd.DataFrame(scores)


def sample_composition(cell_meta: pd.DataFrame, compartment: str) -> pd.DataFrame:
    grouping = ["dataset", "patient_id", "disease_status", "oscc_stage"]
    sample_meta = cell_meta[grouping].drop_duplicates().copy()
    subtype_meta = pd.DataFrame({"subtype": sorted(cell_meta["subtype"].unique())})
    sample_meta["_key"] = 1
    subtype_meta["_key"] = 1
    grid = sample_meta.merge(subtype_meta, on="_key").drop(columns="_key")
    counts = cell_meta.groupby(grouping + ["subtype"], dropna=False, observed=True).size().rename("n_subtype").reset_index()
    totals = cell_meta.groupby(grouping, dropna=False).size().rename("n_parent").reset_index()
    out = grid.merge(counts, on=grouping + ["subtype"], how="left")
    out["n_subtype"] = out["n_subtype"].fillna(0).astype(int)
    out = out.merge(totals, on=grouping, how="left")
    out["fraction_within_parent"] = out["n_subtype"] / out["n_parent"]
    out["compartment"] = compartment
    return out


def cluster_robust_group_test(values: np.ndarray, groups: np.ndarray, patients: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    groups = np.asarray(groups, dtype=object)
    patients = np.asarray(patients, dtype=object)
    if len(np.unique(values)) < 2 or len(np.unique(groups)) < 2:
        return 1.0
    design = pd.get_dummies(pd.Series(groups), drop_first=True, dtype=float)
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(values, design).fit(
        cov_type="cluster", cov_kwds={"groups": patients, "use_correction": True}
    )
    restriction = np.zeros((design.shape[1] - 1, design.shape[1]))
    restriction[:, 1:] = np.eye(design.shape[1] - 1)
    return float(model.wald_test(restriction, scalar=True).pvalue)


def cluster_robust_trend_test(values: np.ndarray, ordinal: np.ndarray, patients: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    ordinal = np.asarray(ordinal, dtype=float)
    patients = np.asarray(patients, dtype=object)
    if len(np.unique(values)) < 2:
        return 1.0
    design = sm.add_constant(ordinal, has_constant="add")
    model = sm.OLS(values, design).fit(
        cov_type="cluster", cov_kwds={"groups": patients, "use_correction": True}
    )
    return float(model.pvalues[1])


def adjusted_candidate_association(
    myeloid: np.ndarray,
    fibroblast: np.ndarray,
    groups: np.ndarray,
    patients: np.ndarray,
) -> tuple[float, float]:
    myeloid = np.asarray(myeloid, dtype=float)
    fibroblast = np.asarray(fibroblast, dtype=float)
    group_design = pd.get_dummies(pd.Series(groups), drop_first=True, dtype=float)
    design = pd.concat([pd.Series(myeloid, name="myeloid"), group_design], axis=1)
    design = sm.add_constant(design, has_constant="add")
    model = sm.OLS(fibroblast, design).fit(
        cov_type="cluster", cov_kwds={"groups": np.asarray(patients), "use_correction": True}
    )
    return float(model.params["myeloid"]), float(model.pvalues["myeloid"])


def stage_tests(composition: pd.DataFrame, compartment: str) -> pd.DataFrame:
    rows = []
    eligible = composition[composition["n_parent"] >= 30].copy()
    for subtype in sorted(eligible["subtype"].unique()):
        sub = eligible[eligible["subtype"] == subtype]
        primary = sub[
            (sub["dataset"] == "A4_GSE181919")
            & sub["disease_status"].isin(["Normal", "OLK", "OSCC"])
        ]
        groups = [
            primary.loc[primary["disease_status"] == group, "fraction_within_parent"].to_numpy()
            for group in ["Normal", "OLK", "OSCC"]
        ]
        if all(len(values) >= 2 for values in groups):
            transformed = np.arcsin(np.sqrt(primary["fraction_within_parent"].to_numpy()))
            p_value = cluster_robust_group_test(
                transformed,
                primary["disease_status"].to_numpy(),
                primary["patient_id"].to_numpy(),
            )
            trend_x = primary["disease_status"].map({"Normal": 0, "OLK": 1, "OSCC": 2}).to_numpy()
            rho, _ = safe_spearman(trend_x, primary["fraction_within_parent"])
            trend_p = cluster_robust_trend_test(
                transformed, trend_x, primary["patient_id"].to_numpy()
            )
            rows.append({
                "compartment": compartment,
                "comparison": "GSE181919 Normal-OLK-OSCC",
                "subtype": subtype,
                "n_group_1": len(groups[0]), "n_group_2": len(groups[1]), "n_group_3": len(groups[2]),
                "group_1": "Normal", "group_2": "OLK", "group_3": "OSCC",
                "median_group_1": float(np.median(groups[0])),
                "median_group_2": float(np.median(groups[1])),
                "median_group_3": float(np.median(groups[2])),
                "effect_last_minus_first": float(np.median(groups[2]) - np.median(groups[0])),
                "p_value": p_value,
                "ordinal_spearman_rho": rho,
                "ordinal_trend_p": trend_p,
            })
        validation = sub[
            (sub["dataset"] == "B5_GSE310797")
            & sub["oscc_stage"].isin(["Early-stage", "Late-stage"])
        ]
        early = validation.loc[validation["oscc_stage"] == "Early-stage", "fraction_within_parent"].to_numpy()
        late = validation.loc[validation["oscc_stage"] == "Late-stage", "fraction_within_parent"].to_numpy()
        if len(early) >= 2 and len(late) >= 2:
            p_value = mannwhitneyu(early, late, alternative="two-sided").pvalue
            rows.append({
                "compartment": compartment,
                "comparison": "GSE310797 Early-Late",
                "subtype": subtype,
                "n_group_1": len(early), "n_group_2": len(late), "n_group_3": np.nan,
                "group_1": "Early-stage", "group_2": "Late-stage", "group_3": "",
                "median_group_1": float(np.median(early)),
                "median_group_2": float(np.median(late)),
                "median_group_3": np.nan,
                "effect_last_minus_first": float(np.median(late) - np.median(early)),
                "p_value": p_value,
                "ordinal_spearman_rho": np.nan,
                "ordinal_trend_p": np.nan,
            })
    out = pd.DataFrame(rows)
    if len(out):
        out["fdr"] = np.nan
        out["trend_fdr"] = np.nan
        for comparison, index in out.groupby("comparison").groups.items():
            out.loc[index, "fdr"] = multipletests(out.loc[index, "p_value"], method="fdr_bh")[1]
            valid = out.loc[index, "ordinal_trend_p"].notna()
            trend_index = out.loc[index].index[valid]
            if len(trend_index):
                out.loc[trend_index, "trend_fdr"] = multipletests(
                    out.loc[trend_index, "ordinal_trend_p"], method="fdr_bh"
                )[1]
    return out


def choose_candidate(tests: pd.DataFrame, allowed: list[str]) -> str:
    primary = tests[
        (tests["comparison"] == "GSE181919 Normal-OLK-OSCC")
        & tests["subtype"].isin(allowed)
    ].copy()
    if primary.empty:
        return allowed[0]
    primary["positive"] = primary["effect_last_minus_first"] > 0
    primary = primary.sort_values(
        ["positive", "trend_fdr", "fdr", "effect_last_minus_first"],
        ascending=[False, True, True, False],
        na_position="last",
    )
    return str(primary.iloc[0]["subtype"])


def subtype_dot_stats(matrix: sparse.csr_matrix, genes: np.ndarray, subtypes: np.ndarray, display_genes: list[str]) -> pd.DataFrame:
    gene_index = {gene: index for index, gene in enumerate(genes)}
    rows = []
    for subtype in sorted(set(subtypes)):
        mask = subtypes == subtype
        for gene in display_genes:
            if gene not in gene_index:
                continue
            values = matrix[mask, gene_index[gene]]
            rows.append({
                "subtype": subtype,
                "gene": gene,
                "mean_expression": float(values.mean()),
                "pct_expressed": float(values.getnnz() / max(mask.sum(), 1) * 100),
            })
    return pd.DataFrame(rows)


def save_figure(fig: plt.Figure, filename: str) -> None:
    FIGURE.mkdir(parents=True, exist_ok=True)
    PDF_FIGURE.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE / f"{filename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(PDF_FIGURE / f"{filename}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_compartment(
    name: str,
    cell_meta: pd.DataFrame,
    embedding_index: np.ndarray,
    embedding: np.ndarray,
    dot_stats: pd.DataFrame,
    composition: pd.DataFrame,
    tests: pd.DataFrame,
    candidate: str,
) -> None:
    subtype_order = sorted(cell_meta["subtype"].unique())
    colors = {subtype: PALETTE[i % len(PALETTE)] for i, subtype in enumerate(subtype_order)}
    fig = plt.figure(figsize=(15.5, 11.8))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.35], height_ratios=[1.0, 0.85], wspace=0.28, hspace=0.52)

    ax = fig.add_subplot(grid[0, 0])
    plotted = cell_meta.iloc[embedding_index]
    plotted_labels = np.asarray([
        "Excluded/contaminant" if subtype.startswith("Excluded") else subtype
        for subtype in plotted["subtype"].to_numpy()
    ])
    plot_order = sorted(set(plotted_labels), key=lambda value: (value == "Excluded/contaminant", value))
    plot_colors = {
        subtype: ("#B8B8B8" if subtype == "Excluded/contaminant" else PALETTE[i % len(PALETTE)])
        for i, subtype in enumerate(plot_order)
    }
    for subtype in plot_order:
        mask = plotted_labels == subtype
        ax.scatter(embedding[mask, 0], embedding[mask, 1], s=2.2, alpha=0.7, color=plot_colors[subtype], label=subtype, rasterized=True)
    ax.set_title(f"A  {name} subtypes")
    for dataset, dataset_label in DATASET_LABELS.items():
        mask = plotted["dataset"].to_numpy() == dataset
        if mask.any():
            ax.text(
                np.median(embedding[mask, 0]), np.quantile(embedding[mask, 1], 0.99) + 0.18,
                dataset_label, ha="center", va="bottom", fontsize=8, weight="bold",
            )
    ax.set_xticks([]); ax.set_yticks([])
    ax.spines[:].set_visible(False)
    ax.legend(
        frameon=False, fontsize=6.2, markerscale=3, ncol=3,
        bbox_to_anchor=(0.5, -0.05), loc="upper center", columnspacing=0.8, handletextpad=0.35,
    )

    ax = fig.add_subplot(grid[0, 1])
    display_subtypes = [subtype for subtype in subtype_order if not subtype.startswith("Excluded")]
    stats = dot_stats[dot_stats["subtype"].isin(display_subtypes)].copy()
    gene_order = list(dict.fromkeys(stats["gene"].tolist()))
    mean = stats.pivot(index="gene", columns="subtype", values="mean_expression").reindex(index=gene_order, columns=display_subtypes)
    pct = stats.pivot(index="gene", columns="subtype", values="pct_expressed").reindex(index=gene_order, columns=display_subtypes)
    scaled = mean.copy()
    for gene in gene_order:
        values = mean.loc[gene].to_numpy(float)
        scaled.loc[gene] = (values - np.nanmean(values)) / (np.nanstd(values) + 1e-6)
    for y, gene in enumerate(gene_order):
        for x, subtype in enumerate(display_subtypes):
            ax.scatter(x, y, s=10 + 1.4 * pct.loc[gene, subtype], c=scaled.loc[gene, subtype], cmap="RdBu_r", vmin=-1.5, vmax=2.5, edgecolors="none")
    ax.set_xticks(range(len(display_subtypes)), display_subtypes, rotation=40, ha="right", fontsize=7)
    ax.set_yticks(range(len(gene_order)), gene_order, fontsize=7)
    ax.invert_yaxis(); ax.set_title("B  Literature-marker audit")
    ax.grid(color="#E8E8E8", linewidth=0.4)

    ax = fig.add_subplot(grid[1, 0])
    primary = composition[
        (composition["dataset"] == "A4_GSE181919")
        & composition["disease_status"].isin(["Normal", "OLK", "OSCC"])
        & (composition["n_parent"] >= 30)
        & (~composition["subtype"].str.startswith("Excluded"))
    ]
    median = primary.groupby(["subtype", "disease_status"])["fraction_within_parent"].median().unstack(fill_value=0)
    median = median.reindex(index=[x for x in subtype_order if x in median.index], columns=["Normal", "OLK", "OSCC"], fill_value=0)
    image = ax.imshow(median.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(0.1, float(np.nanquantile(median, 0.95))))
    ax.set_xticks(range(3), ["Normal", "OLK", "OSCC"])
    ax.set_yticks(range(len(median)), median.index, fontsize=7)
    ax.set_title("C  GSE181919 patient-median subtype fractions")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    colorbar.set_label("Median fraction within parent", fontsize=8)

    ax = fig.add_subplot(grid[1, 1])
    primary_candidate = primary[primary["subtype"] == candidate]
    for x, group in enumerate(["Normal", "OLK", "OSCC"]):
        values = primary_candidate.loc[primary_candidate["disease_status"] == group, "fraction_within_parent"].to_numpy()
        jitter = RNG.normal(0, 0.045, len(values))
        ax.scatter(np.full(len(values), x) + jitter, values, color=GROUP_COLORS[group], alpha=0.8, s=28, edgecolor="white", linewidth=0.4)
        if len(values):
            ax.plot([x - 0.2, x + 0.2], [np.median(values)] * 2, color="#222222", linewidth=2)
    test = tests[(tests["comparison"] == "GSE181919 Normal-OLK-OSCC") & (tests["subtype"] == candidate)]
    fdr = float(test.iloc[0]["fdr"]) if len(test) else np.nan
    ax.set_xticks(range(3), ["Normal", "OLK", "OSCC"])
    ax.set_ylabel("Fraction within parent compartment")
    ax.set_title(f"D  Candidate: {candidate}\npatient-level FDR={fdr:.3g}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"Literature-v3 {name.lower()} state screen", fontsize=14, weight="bold")
    save_figure(fig, f"Fig3_{name.lower()}_state_screen_literature_v3")


def analyze_compartment(name: str, config: dict, stage_map: dict[str, str], sample_totals: pd.DataFrame) -> dict:
    print(f"\n[{name}] loading parent compartment...", flush=True)
    backed = ad.read_h5ad(H5AD, backed="r")
    parent_mask = backed.obs[CELL_TYPE_FIELD].astype(str).to_numpy() == config["parent"]
    obs = backed.obs.loc[parent_mask, ["dataset", "disease_status", "patient_id", "author_cell_type"]].copy()
    genes = backed.var_names.astype(str).to_numpy()
    matrix = backed.X[parent_mask, :].tocsr().astype(np.float32)
    backed.file.close()
    for column in obs.columns:
        obs[column] = obs[column].astype(str)
    datasets = obs["dataset"].to_numpy()
    print(f"[{name}] {matrix.shape[0]:,} cells x {matrix.shape[1]:,} genes", flush=True)

    forced = {gene for values in config["programs"].values() for gene in values}
    forced.update(gene for values in CONTAMINANT_PROGRAMS.values() for gene in values)
    subclusters = np.empty(len(obs), dtype=object)
    subtypes = np.empty(len(obs), dtype=object)
    annotation_frames = []
    marker_frames = []
    score_frames = []
    hvg_frames = []
    embedding_indices = []
    embedding_parts = []
    variance_values = []
    edge_total = 0
    cluster_total = 0

    for dataset_number, dataset in enumerate(sorted(set(datasets))):
        global_idx = np.flatnonzero(datasets == dataset)
        local_matrix = matrix[global_idx]
        local_datasets = np.repeat(dataset, len(global_idx))
        print(f"[{name}/{DATASET_LABELS[dataset]}] selecting HVGs for {len(global_idx):,} cells...", flush=True)
        hvg_idx = choose_hvgs(local_matrix, genes, local_datasets, forced)
        print(f"[{name}/{DATASET_LABELS[dataset]}] sparse SVD on {len(hvg_idx):,} genes...", flush=True)
        pcs, variance = batch_center_and_pca(local_matrix, hvg_idx, local_datasets)
        print(f"[{name}/{DATASET_LABELS[dataset]}] kNN graph and Louvain...", flush=True)
        local_clusters, n_edges = graph_cluster(pcs, config["resolution"])
        n_clusters = len(set(local_clusters))
        print(f"[{name}/{DATASET_LABELS[dataset]}] {n_clusters} clusters; {n_edges:,} edges", flush=True)
        cluster_total += n_clusters
        edge_total += n_edges
        variance_values.append((len(global_idx), float(variance.sum())))

        cluster_means, _, raw_marker_rows = cluster_expression(local_matrix, local_clusters)
        lineage_scores = program_scores_by_cluster(cluster_means, genes, config["programs"])
        contaminant_scores = program_scores_by_cluster(cluster_means, genes, CONTAMINANT_PROGRAMS)
        local_annotation = annotate_clusters(lineage_scores, contaminant_scores, config["programs"])
        cluster_sizes = pd.Series(local_clusters).value_counts().sort_index()
        local_annotation["n_cells"] = local_annotation["subcluster"].astype(int).map(cluster_sizes)
        prefix = DATASET_LABELS[dataset]
        local_annotation["local_cluster"] = local_annotation["subcluster"]
        local_annotation["subcluster"] = prefix + "_" + local_annotation["subcluster"].astype(str)
        local_annotation["dataset"] = dataset
        local_annotation["dataset_label"] = prefix
        annotation_frames.append(local_annotation)

        local_map = local_annotation.set_index("local_cluster")["subtype"].to_dict()
        local_types = np.asarray([local_map[str(cluster)] for cluster in local_clusters], dtype=object)
        local_ids = np.asarray([f"{prefix}_{cluster}" for cluster in local_clusters], dtype=object)
        subclusters[global_idx] = local_ids
        subtypes[global_idx] = local_types

        local_markers = marker_table(raw_marker_rows, genes, local_annotation.rename(
            columns={"subcluster": "prefixed_subcluster", "local_cluster": "subcluster"}
        ))
        local_markers["local_cluster"] = local_markers["subcluster"]
        local_markers["subcluster"] = prefix + "_" + local_markers["subcluster"].astype(str)
        local_markers["dataset"] = dataset
        marker_frames.append(local_markers)

        local_scores = lineage_scores.copy()
        local_scores["local_cluster"] = local_scores["subcluster"]
        local_scores["subcluster"] = prefix + "_" + local_scores["subcluster"].astype(str)
        local_scores["dataset"] = dataset
        local_scores = local_scores.merge(
            local_annotation[["subcluster", "subtype"]], on="subcluster", how="left"
        )
        score_frames.append(local_scores)
        hvg_frames.append(pd.DataFrame({"dataset": dataset, "gene": genes[hvg_idx]}))

        print(f"[{name}/{DATASET_LABELS[dataset]}] display embedding...", flush=True)
        local_embedding_idx, local_embedding = stratified_tsne(pcs, local_clusters, max_cells=5000)
        local_embedding -= np.median(local_embedding, axis=0, keepdims=True)
        scale = np.quantile(np.abs(local_embedding), 0.9, axis=0)
        local_embedding /= np.where(scale > 0, scale, 1.0)
        local_embedding[:, 0] += dataset_number * 3.2
        embedding_indices.append(global_idx[local_embedding_idx])
        embedding_parts.append(local_embedding)

    annotation = pd.concat(annotation_frames, ignore_index=True)
    markers = pd.concat(marker_frames, ignore_index=True)
    lineage_scores = pd.concat(score_frames, ignore_index=True)
    hvg_table = pd.concat(hvg_frames, ignore_index=True)
    embedding_index = np.concatenate(embedding_indices)
    embedding = np.vstack(embedding_parts)
    print(f"[{name}] {cluster_total} within-dataset clusters; {edge_total:,} total graph edges", flush=True)

    cell_meta = obs.reset_index(names="cell_id")
    cell_meta["subcluster"] = subclusters
    cell_meta["subtype"] = subtypes
    cell_meta["oscc_stage"] = cell_meta["patient_id"].astype(str).map(stage_map).fillna("")
    cell_meta["compartment"] = name
    program_cell = per_cell_program_scores(matrix, genes, config["programs"])
    for column in program_cell:
        cell_meta[f"program_{column}"] = program_cell[column].to_numpy()

    composition = sample_composition(cell_meta, name)
    composition = composition.merge(sample_totals, on=["dataset", "patient_id"], how="left")
    composition["fraction_of_all_cells"] = composition["n_subtype"] / composition["n_all_cells"]
    tests = stage_tests(composition, name)
    candidate = choose_candidate(tests, config["candidate_labels"])
    dot_stats = subtype_dot_stats(matrix, genes, subtypes, config["display_genes"])

    prefix = name.lower()
    cell_meta.to_csv(RESULT / f"{prefix}_subcluster_cells_literature_v3.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    annotation.to_csv(RESULT / f"{prefix}_subcluster_annotation_literature_v3.csv", index=False, encoding="utf-8-sig")
    lineage_scores.to_csv(
        RESULT / f"{prefix}_subcluster_program_scores_literature_v3.csv", index=False, encoding="utf-8-sig"
    )
    markers.to_csv(RESULT / f"{prefix}_subcluster_markers_literature_v3.csv.gz", index=False, compression="gzip", encoding="utf-8-sig")
    dot_stats.to_csv(RESULT / f"{prefix}_subtype_marker_audit_literature_v3.csv", index=False, encoding="utf-8-sig")
    composition.to_csv(RESULT / f"{prefix}_subtype_sample_composition_literature_v3.csv", index=False, encoding="utf-8-sig")
    tests.to_csv(RESULT / f"{prefix}_subtype_stage_tests_literature_v3.csv", index=False, encoding="utf-8-sig")
    hvg_table.to_csv(RESULT / f"{prefix}_subcluster_hvgs_literature_v3.csv", index=False)
    embedding_table = cell_meta.iloc[embedding_index][
        ["cell_id", "dataset", "disease_status", "patient_id", "subcluster", "subtype"]
    ].copy()
    embedding_table["embedding_x"] = embedding[:, 0]
    embedding_table["embedding_y"] = embedding[:, 1]
    embedding_table.to_csv(
        RESULT / f"{prefix}_subcluster_display_embedding_literature_v3.csv.gz",
        index=False, compression="gzip", encoding="utf-8-sig",
    )

    plot_compartment(name, cell_meta, embedding_index, embedding, dot_stats, composition, tests, candidate)
    return {
        "name": name,
        "candidate": candidate,
        "cell_meta": cell_meta,
        "composition": composition,
        "tests": tests,
        "annotation": annotation,
        "n_cells": int(matrix.shape[0]),
        "n_clusters": int(cluster_total),
        "n_hvgs": int(hvg_table["gene"].nunique()),
        "pca_variance_30": float(sum(n * value for n, value in variance_values) / sum(n for n, _ in variance_values)),
    }


def build_niche(myeloid: dict, fibroblast: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    def candidate_table(result: dict) -> pd.DataFrame:
        data = result["composition"]
        return data[data["subtype"] == result["candidate"]][
            ["dataset", "patient_id", "disease_status", "oscc_stage", "n_parent", "fraction_within_parent", "fraction_of_all_cells"]
        ].rename(columns={
            "n_parent": f"n_{result['name'].lower()}_parent",
            "fraction_within_parent": f"{result['name'].lower()}_candidate_within_parent",
            "fraction_of_all_cells": f"{result['name'].lower()}_candidate_of_all_cells",
        })

    niche = candidate_table(myeloid).merge(
        candidate_table(fibroblast),
        on=["dataset", "patient_id", "disease_status", "oscc_stage"],
        how="inner",
    )
    for dataset in niche["dataset"].unique():
        mask = niche["dataset"] == dataset
        for source in ["myeloid_candidate_within_parent", "fibroblast_candidate_within_parent"]:
            values = niche.loc[mask, source]
            niche.loc[mask, f"z_{source}"] = (values - values.mean()) / (values.std(ddof=0) + 1e-8)
    niche["joint_state_score"] = niche["z_myeloid_candidate_within_parent"] + niche["z_fibroblast_candidate_within_parent"]
    niche["joint_abundance_geomean"] = np.sqrt(
        niche["myeloid_candidate_of_all_cells"] * niche["fibroblast_candidate_of_all_cells"]
    )

    rows = []
    primary = niche[
        (niche["dataset"] == "A4_GSE181919")
        & niche["disease_status"].isin(["Normal", "OLK", "OSCC"])
        & (niche["n_myeloid_parent"] >= 30)
        & (niche["n_fibroblast_parent"] >= 30)
    ].copy()
    rho, p_value = safe_spearman(
        primary["myeloid_candidate_within_parent"],
        primary["fibroblast_candidate_within_parent"],
    )
    groups = [primary.loc[primary["disease_status"] == group, "joint_state_score"].to_numpy() for group in ["Normal", "OLK", "OSCC"]]
    group_p = cluster_robust_group_test(
        primary["joint_state_score"].to_numpy(),
        primary["disease_status"].to_numpy(),
        primary["patient_id"].to_numpy(),
    ) if all(len(values) >= 2 for values in groups) else np.nan
    adjusted_beta, adjusted_p = adjusted_candidate_association(
        primary["myeloid_candidate_within_parent"].to_numpy(),
        primary["fibroblast_candidate_within_parent"].to_numpy(),
        primary["disease_status"].to_numpy(),
        primary["patient_id"].to_numpy(),
    )
    rows.append({
        "comparison": "GSE181919 Normal-OLK-OSCC",
        "myeloid_candidate": myeloid["candidate"],
        "fibroblast_candidate": fibroblast["candidate"],
        "n_samples": len(primary),
        "candidate_fraction_spearman_rho": rho,
        "candidate_fraction_correlation_p": p_value,
        "stage_adjusted_association_beta": adjusted_beta,
        "stage_adjusted_association_p": adjusted_p,
        "joint_state_group_p": group_p,
        "median_joint_Normal": np.median(groups[0]) if len(groups[0]) else np.nan,
        "median_joint_OLK": np.median(groups[1]) if len(groups[1]) else np.nan,
        "median_joint_OSCC": np.median(groups[2]) if len(groups[2]) else np.nan,
    })
    validation = niche[
        (niche["dataset"] == "B5_GSE310797")
        & niche["oscc_stage"].isin(["Early-stage", "Late-stage"])
        & (niche["n_myeloid_parent"] >= 30)
        & (niche["n_fibroblast_parent"] >= 30)
    ].copy()
    validation_rho, validation_corr_p = safe_spearman(
        validation["myeloid_candidate_within_parent"],
        validation["fibroblast_candidate_within_parent"],
    )
    early = validation.loc[validation["oscc_stage"] == "Early-stage", "joint_state_score"].to_numpy()
    late = validation.loc[validation["oscc_stage"] == "Late-stage", "joint_state_score"].to_numpy()
    rows.append({
        "comparison": "GSE310797 Early-Late",
        "myeloid_candidate": myeloid["candidate"],
        "fibroblast_candidate": fibroblast["candidate"],
        "n_samples": len(validation),
        "candidate_fraction_spearman_rho": validation_rho,
        "candidate_fraction_correlation_p": validation_corr_p,
        "stage_adjusted_association_beta": adjusted_candidate_association(
            validation["myeloid_candidate_within_parent"].to_numpy(),
            validation["fibroblast_candidate_within_parent"].to_numpy(),
            validation["oscc_stage"].to_numpy(),
            validation["patient_id"].to_numpy(),
        )[0] if len(validation) >= 4 else np.nan,
        "stage_adjusted_association_p": adjusted_candidate_association(
            validation["myeloid_candidate_within_parent"].to_numpy(),
            validation["fibroblast_candidate_within_parent"].to_numpy(),
            validation["oscc_stage"].to_numpy(),
            validation["patient_id"].to_numpy(),
        )[1] if len(validation) >= 4 else np.nan,
        "joint_state_group_p": mannwhitneyu(early, late, alternative="two-sided").pvalue if len(early) >= 2 and len(late) >= 2 else np.nan,
        "median_joint_Normal": np.nan,
        "median_joint_OLK": np.median(early) if len(early) else np.nan,
        "median_joint_OSCC": np.median(late) if len(late) else np.nan,
    })
    return niche, pd.DataFrame(rows)


def plot_niche(niche: pd.DataFrame, summary: pd.DataFrame, myeloid_candidate: str, fibroblast_candidate: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
    primary = niche[
        (niche["dataset"] == "A4_GSE181919")
        & niche["disease_status"].isin(["Normal", "OLK", "OSCC"])
        & (niche["n_myeloid_parent"] >= 30)
        & (niche["n_fibroblast_parent"] >= 30)
    ]
    for group in ["Normal", "OLK", "OSCC"]:
        sub = primary[primary["disease_status"] == group]
        axes[0].scatter(
            sub["myeloid_candidate_within_parent"], sub["fibroblast_candidate_within_parent"],
            color=GROUP_COLORS[group], label=group, s=42, alpha=0.85, edgecolor="white", linewidth=0.5,
        )
    row = summary[summary["comparison"] == "GSE181919 Normal-OLK-OSCC"].iloc[0]
    axes[0].set_xlabel(f"{myeloid_candidate}\nwithin Myeloid")
    axes[0].set_ylabel(f"{fibroblast_candidate}\nwithin Fibroblasts")
    axes[0].set_title(
        f"A  Candidate co-variation\nSpearman r={row['candidate_fraction_spearman_rho']:.2f}; "
        f"stage-adjusted p={row['stage_adjusted_association_p']:.3g}"
    )
    axes[0].legend(frameon=False)

    for x, group in enumerate(["Normal", "OLK", "OSCC"]):
        values = primary.loc[primary["disease_status"] == group, "joint_state_score"].to_numpy()
        axes[1].scatter(np.full(len(values), x) + RNG.normal(0, 0.05, len(values)), values, color=GROUP_COLORS[group], s=34, alpha=0.85)
        if len(values): axes[1].plot([x - 0.2, x + 0.2], [np.median(values)] * 2, color="#222222", linewidth=2)
    axes[1].set_xticks(range(3), ["Normal", "OLK", "OSCC"])
    axes[1].set_ylabel("Joint within-parent state score")
    axes[1].set_title(f"B  Discovery cohort\npatient-level p={row['joint_state_group_p']:.3g}")

    validation = niche[
        (niche["dataset"] == "B5_GSE310797")
        & niche["oscc_stage"].isin(["Early-stage", "Late-stage"])
        & (niche["n_myeloid_parent"] >= 30)
        & (niche["n_fibroblast_parent"] >= 30)
    ]
    for x, group in enumerate(["Early-stage", "Late-stage"]):
        values = validation.loc[validation["oscc_stage"] == group, "joint_state_score"].to_numpy()
        axes[2].scatter(np.full(len(values), x) + RNG.normal(0, 0.05, len(values)), values, color=STAGE_COLORS[group], s=34, alpha=0.85)
        if len(values): axes[2].plot([x - 0.2, x + 0.2], [np.median(values)] * 2, color="#222222", linewidth=2)
    row = summary[summary["comparison"] == "GSE310797 Early-Late"].iloc[0]
    axes[2].set_xticks(range(2), ["Early", "Late"])
    axes[2].set_ylabel("Joint within-parent state score")
    axes[2].set_title(f"C  Stage validation\npatient-level p={row['joint_state_group_p']:.3g}")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Exploratory myeloid-fibroblast niche decision gate", fontsize=13, weight="bold")
    fig.tight_layout()
    save_figure(fig, "Fig3_candidate_niche_decision_literature_v3")


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    stage = pd.read_csv(RESULT / "literature_v3_GSE310797_confirmed_stage_mapping.csv")
    stage_map = stage.set_index("patient_id")["stage_group"].astype(str).to_dict()

    backed = ad.read_h5ad(H5AD, backed="r")
    totals = backed.obs.groupby(["dataset", "patient_id"], observed=True).size().rename("n_all_cells").reset_index()
    totals["dataset"] = totals["dataset"].astype(str)
    totals["patient_id"] = totals["patient_id"].astype(str)
    backed.file.close()

    results = {}
    for name, config in CONFIGS.items():
        results[name] = analyze_compartment(name, config, stage_map, totals)

    all_tests = pd.concat([results["Myeloid"]["tests"], results["Fibroblast"]["tests"]], ignore_index=True)
    all_tests.to_csv(RESULT / "tam_caf_subtype_stage_tests_literature_v3.csv", index=False, encoding="utf-8-sig")
    niche, niche_summary = build_niche(results["Myeloid"], results["Fibroblast"])
    niche.to_csv(RESULT / "tam_caf_candidate_niche_sample_scores_literature_v3.csv", index=False, encoding="utf-8-sig")
    niche_summary.to_csv(RESULT / "tam_caf_candidate_niche_tests_literature_v3.csv", index=False, encoding="utf-8-sig")
    plot_niche(
        niche, niche_summary, results["Myeloid"]["candidate"], results["Fibroblast"]["candidate"]
    )

    my_primary = all_tests[
        (all_tests["comparison"] == "GSE181919 Normal-OLK-OSCC")
        & (all_tests["compartment"] == "Myeloid")
        & (all_tests["subtype"] == results["Myeloid"]["candidate"])
    ].iloc[0]
    fibro_primary = all_tests[
        (all_tests["comparison"] == "GSE181919 Normal-OLK-OSCC")
        & (all_tests["compartment"] == "Fibroblast")
        & (all_tests["subtype"] == results["Fibroblast"]["candidate"])
    ].iloc[0]
    discovery_niche = niche_summary[niche_summary["comparison"] == "GSE181919 Normal-OLK-OSCC"].iloc[0]
    validation_niche = niche_summary[niche_summary["comparison"] == "GSE310797 Early-Late"].iloc[0]
    advance = (
        my_primary["fdr"] < 0.05
        and fibro_primary["fdr"] < 0.05
        and discovery_niche["stage_adjusted_association_p"] < 0.05
    )
    decision = {
        "analysis_status": "advance_provisional_niche_to_mechanistic_and_spatial_validation" if advance else "hold_niche_hypothesis",
        "myeloid_candidate": results["Myeloid"]["candidate"],
        "fibroblast_candidate": results["Fibroblast"]["candidate"],
        "discovery_evidence": {
            "myeloid_candidate_fdr": float(my_primary["fdr"]),
            "fibroblast_candidate_fdr": float(fibro_primary["fdr"]),
            "stage_adjusted_candidate_association_p": float(discovery_niche["stage_adjusted_association_p"]),
            "joint_state_group_p": float(discovery_niche["joint_state_group_p"]),
        },
        "validation_evidence": {
            "GSE310797_joint_state_early_late_p": float(validation_niche["joint_state_group_p"]),
            "interpretation": "directionally supportive trend; not statistically confirmed",
        },
        "myeloid_cells": results["Myeloid"]["n_cells"],
        "fibroblast_cells": results["Fibroblast"]["n_cells"],
        "myeloid_clusters": results["Myeloid"]["n_clusters"],
        "fibroblast_clusters": results["Fibroblast"]["n_clusters"],
        "references": {
            "GSE181919": "Choi et al. 2023; doi:10.1038/s41467-023-36691-x",
            "GSE310797": "Li et al. 2026; doi:10.1038/s41467-026-70861-x",
            "GSE164241": "Williams et al. 2021; doi:10.1016/j.cell.2021.05.013; Tables S4/S5",
        },
        "caveat": "Subtype labels are literature-guided reconstructions and candidates were selected in the discovery cohort; independent spatial/OED validation is required before a definitive niche claim.",
    }
    with open(RESULT / "tam_caf_subclustering_decision_literature_v3.json", "w", encoding="utf-8") as handle:
        json.dump(decision, handle, ensure_ascii=False, indent=2)
    print("\nTAM/CAF subclustering decision gate complete.", flush=True)
    print(json.dumps(decision, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
