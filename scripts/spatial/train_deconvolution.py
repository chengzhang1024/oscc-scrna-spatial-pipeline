from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
import scvi
import torch
from cell2location.models import Cell2location, RegressionModel
from scipy import sparse
from scvi.model import CondSCVI, DestVI


SPATIAL_PATH = (
    ROOT
    / "data"
    / "intermediate"
    / "GSE208253"
    / "GSE208253_cleaned_pre_deconvolution.h5ad"
)
REFERENCE_PATH = ROOT / "data" / "raw" / "GSE181919" / "GSE181919.h5ad"
FINAL_PATH = ROOT / "data" / "processed" / "GSE208253" / "GSE208253_cleaned_deconvolved.h5ad"
RESULT_DIR = ROOT / "results" / "spatial_preprocessing"
C2L_ROOT = ROOT / "models" / "cell2location"
DESTVI_ROOT = ROOT / "models" / "destvi"

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

AUTHOR_TO_NINE = {
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
}


def ensure_counts(adata: ad.AnnData, layer: str | None = None) -> None:
    matrix = adata.layers[layer] if layer else adata.X
    if not sparse.issparse(matrix):
        matrix = sparse.csr_matrix(matrix)
    matrix = matrix.tocsr()
    sample = matrix.data[: min(matrix.data.size, 100000)]
    if sample.size and not np.allclose(sample, np.round(sample)):
        raise ValueError("Model input is not raw integer counts")


def stratified_reference_subset(adata: ad.AnnData, n_per_type: int = 100) -> ad.AnnData:
    rng = np.random.default_rng(20260727)
    selected: list[int] = []
    labels = adata.obs["cell_type"].to_numpy()
    for cell_type in CELL_TYPES:
        indices = np.flatnonzero(labels == cell_type)
        selected.extend(rng.choice(indices, size=min(n_per_type, len(indices)), replace=False).tolist())
    return adata[np.sort(selected)].copy()


def load_reference(smoke_test: bool) -> ad.AnnData:
    reference = sc.read_h5ad(REFERENCE_PATH)
    if "original_celltype" not in reference.obs:
        raise KeyError("GSE181919 original_celltype is missing")
    reference.obs["cell_type"] = reference.obs["original_celltype"].map(AUTHOR_TO_NINE)
    if reference.obs["cell_type"].isna().any():
        unknown = reference.obs.loc[reference.obs["cell_type"].isna(), "original_celltype"].unique()
        raise ValueError(f"Unmapped original labels: {unknown}")
    reference.obs["cell_type"] = pd.Categorical(
        reference.obs["cell_type"], categories=CELL_TYPES, ordered=True
    )
    reference.obs["sample_id"] = reference.obs["sample_id"].astype(str)
    reference.var_names_make_unique()
    ensure_counts(reference)
    if smoke_test:
        reference = stratified_reference_subset(reference)
    return reference


def select_regression_genes(reference: ad.AnnData, spatial: ad.AnnData, smoke_test: bool) -> list[str]:
    n_cells = np.asarray((reference.X > 0).sum(axis=0)).ravel()
    sums = np.asarray(reference.X.sum(axis=0)).ravel()
    nonzero_mean = np.divide(sums, n_cells, out=np.zeros_like(sums, dtype=float), where=n_cells > 0)
    selected = (n_cells >= reference.n_obs * 0.001) | ((n_cells >= 5) & (nonzero_mean >= 1.0))
    genes = reference.var_names[selected & reference.var_names.isin(spatial.var_names)].tolist()
    if smoke_test:
        genes = genes[: min(500, len(genes))]
    if len(genes) < 100:
        raise RuntimeError(f"Too few common regression genes: {len(genes)}")
    return genes


def select_destvi_genes(reference: ad.AnnData, spatial: ad.AnnData, smoke_test: bool) -> list[str]:
    ref_n = np.asarray((reference.X > 0).sum(axis=0)).ravel()
    spatial_counts = spatial.layers["counts"]
    st_n = np.asarray((spatial_counts > 0).sum(axis=0)).ravel()
    ref_keep = pd.Series(ref_n >= 10, index=reference.var_names)
    st_keep = pd.Series(st_n >= 10, index=spatial.var_names)
    genes = reference.var_names[
        reference.var_names.isin(st_keep.index[st_keep]) & ref_keep.to_numpy()
    ].tolist()
    if smoke_test:
        genes = genes[: min(500, len(genes))]
    if len(genes) < 100:
        raise RuntimeError(f"Too few common DestVI genes: {len(genes)}")
    return genes


def train_cell2location(
    reference: ad.AnnData,
    spatial: ad.AnnData,
    genes: list[str],
    smoke_test: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("Training cell2location reference regression", flush=True)
    ref = reference[:, genes].copy()
    RegressionModel.setup_anndata(
        adata=ref,
        batch_key="sample_id",
        labels_key="cell_type",
    )
    regression = RegressionModel(ref)
    regression.train(
        max_epochs=2 if smoke_test else 250,
        batch_size=min(2500, ref.n_obs),
        enable_progress_bar=smoke_test,
    )
    ref = regression.export_posterior(
        ref,
        sample_kwargs={
            "num_samples": 10 if smoke_test else 500,
            "batch_size": min(2500, ref.n_obs),
        },
    )
    if not smoke_test:
        regression.save(C2L_ROOT / "reference_regression", overwrite=True, save_anndata=True)

    factor_names = list(regression.factor_names_)
    signatures = pd.DataFrame(
        np.asarray(ref.varm["means_per_cluster_mu_fg"]),
        index=ref.var_names,
        columns=factor_names,
    )
    signatures = signatures.reindex(columns=CELL_TYPES)

    st = spatial[:, genes].copy()
    st.X = st.layers["counts"].copy()
    Cell2location.setup_anndata(st, batch_key="sample_id")
    model = Cell2location(
        st,
        cell_state_df=signatures,
        N_cells_per_location=10,
        detection_alpha=20,
    )
    print("Training cell2location spatial abundance model", flush=True)
    model.train(
        max_epochs=2 if smoke_test else 300,
        batch_size=min(2048, st.n_obs),
        train_size=1,
        enable_progress_bar=smoke_test,
    )
    st = model.export_posterior(
        st,
        sample_kwargs={
            "num_samples": 10 if smoke_test else 500,
            "batch_size": min(2048, st.n_obs),
        },
    )
    if not smoke_test:
        model.save(C2L_ROOT / "spatial_model", overwrite=True, save_anndata=True)

    # cell2location prefixes exported DataFrame column names with the posterior
    # variable name. Use the known model factor order instead of label reindexing.
    abundance = pd.DataFrame(
        np.asarray(st.obsm["means_cell_abundance_w_sf"]),
        index=st.obs_names,
        columns=CELL_TYPES,
    )
    q05 = pd.DataFrame(
        np.asarray(st.obsm["q05_cell_abundance_w_sf"]),
        index=st.obs_names,
        columns=CELL_TYPES,
    )
    proportions = abundance.div(abundance.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return abundance, q05, proportions


def train_destvi(
    reference: ad.AnnData,
    spatial: ad.AnnData,
    genes: list[str],
    smoke_test: bool,
) -> pd.DataFrame:
    print("Training DestVI single-cell conditional model", flush=True)
    ref = reference[:, genes].copy()
    CondSCVI.setup_anndata(ref, labels_key="cell_type")
    rna_model = CondSCVI(ref, weight_obs=True)
    rna_model.train(
        max_epochs=2 if smoke_test else 200,
        batch_size=min(1024, ref.n_obs),
        accelerator="gpu",
        devices=1,
        enable_progress_bar=smoke_test,
    )
    if not smoke_test:
        rna_model.save(DESTVI_ROOT / "single_cell_model", overwrite=True, save_anndata=True)

    all_proportions: list[pd.DataFrame] = []
    for sample_id in spatial.obs["sample_id"].astype(str).unique():
        print(f"Training DestVI spatial model for {sample_id}", flush=True)
        st = spatial[spatial.obs["sample_id"].astype(str) == sample_id, genes].copy()
        st.X = st.layers["counts"].copy()
        DestVI.setup_anndata(st)
        spatial_model = DestVI.from_rna_model(st, rna_model)
        spatial_model.train(
            max_epochs=2 if smoke_test else 200,
            batch_size=min(2048, st.n_obs),
            accelerator="gpu",
            devices=1,
            enable_progress_bar=smoke_test,
        )
        proportions = spatial_model.get_proportions(st)
        proportions = proportions.reindex(columns=CELL_TYPES).fillna(0)
        # DestVI excludes its technical noise component by default, so the
        # retained biological labels can sum slightly below one. The requested
        # output contains exactly the nine paper labels; renormalize over them.
        proportions = proportions.div(proportions.sum(axis=1), axis=0)
        all_proportions.append(proportions)
        if not smoke_test:
            st.obsm["proportions"] = proportions
            spatial_model.save(
                DESTVI_ROOT / f"spatial_model_{sample_id}",
                overwrite=True,
                save_anndata=True,
            )
    return pd.concat(all_proportions, axis=0).reindex(spatial.obs_names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()

    torch.set_float32_matmul_precision("high")
    scvi.settings.seed = 20260727
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this workflow")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    C2L_ROOT.mkdir(parents=True, exist_ok=True)
    DESTVI_ROOT.mkdir(parents=True, exist_ok=True)
    spatial = sc.read_h5ad(SPATIAL_PATH)
    ensure_counts(spatial, layer="counts")
    if args.smoke_test:
        smoke_indices: list[int] = []
        sample_values = spatial.obs["sample_id"].astype(str).to_numpy()
        for sample_id in sorted(np.unique(sample_values)):
            smoke_indices.extend(np.flatnonzero(sample_values == sample_id)[:100].tolist())
        spatial = spatial[np.asarray(smoke_indices)].copy()
    reference = load_reference(args.smoke_test)

    label_counts = reference.obs["cell_type"].value_counts(sort=False).reindex(CELL_TYPES)
    print(f"Reference cells: {label_counts.to_dict()}", flush=True)
    if not args.smoke_test:
        label_counts.rename("n_cells").to_csv(RESULT_DIR / "gse181919_reference_label_counts.csv")

    regression_genes = select_regression_genes(reference, spatial, args.smoke_test)
    destvi_genes = select_destvi_genes(reference, spatial, args.smoke_test)
    print(
        f"Genes: cell2location={len(regression_genes)}, DestVI={len(destvi_genes)}",
        flush=True,
    )
    if not args.smoke_test:
        pd.Series(regression_genes, name="gene").to_csv(
            RESULT_DIR / "cell2location_model_genes.csv", index=False
        )
        pd.Series(destvi_genes, name="gene").to_csv(
            RESULT_DIR / "destvi_model_genes.csv", index=False
        )

    abundance, q05, c2l_proportions = train_cell2location(
        reference, spatial, regression_genes, args.smoke_test
    )
    destvi_proportions = train_destvi(reference, spatial, destvi_genes, args.smoke_test)
    if args.smoke_test:
        print("Smoke test completed", flush=True)
        return

    spatial.obsm["cell2location_abundance"] = abundance.reindex(spatial.obs_names)
    spatial.obsm["cell2location_q05"] = q05.reindex(spatial.obs_names)
    spatial.obsm["cell2location_proportions"] = c2l_proportions.reindex(spatial.obs_names)
    spatial.obsm["destvi_proportions"] = destvi_proportions.reindex(spatial.obs_names)
    spatial.obs["dominant_cell_type"] = pd.Categorical(
        c2l_proportions.idxmax(axis=1), categories=CELL_TYPES, ordered=True
    )
    spatial.uns["deconvolution"] = {
        "reference_dataset": "GSE181919",
        "reference_annotation": "Choi et al. 2023 original nine major cell types",
        "cell_types": CELL_TYPES,
        "cell2location": {
            "N_cells_per_location": 10,
            "detection_alpha": 20,
            "abundance_slot": "obsm/cell2location_abundance",
            "proportion_slot": "obsm/cell2location_proportions",
        },
        "destvi": {
            "proportion_slot": "obsm/destvi_proportions",
            "spatial_models": "one model per Visium section to avoid cross-section technical mixing",
            "cell_type_specific_expression": "query trained model with score_spatial_geneset.py",
        },
        "interpretation": "All cell-type-specific spatial expression values are model-inferred, not directly measured single-cell values.",
    }

    qc_rows = []
    for cell_type in CELL_TYPES:
        x = c2l_proportions[cell_type].to_numpy()
        y = destvi_proportions[cell_type].to_numpy()
        qc_rows.append(
            {
                "cell_type": cell_type,
                "cell2location_mean": float(np.mean(x)),
                "destvi_mean": float(np.mean(y)),
                "spearman_r": float(pd.Series(x).corr(pd.Series(y), method="spearman")),
            }
        )
    pd.DataFrame(qc_rows).to_csv(RESULT_DIR / "deconvolution_model_concordance.csv", index=False)

    spatial.write_h5ad(FINAL_PATH, compression="gzip")
    metadata = {
        "final_path": str(FINAL_PATH),
        "n_spots": int(spatial.n_obs),
        "n_genes": int(spatial.n_vars),
        "cell_types": CELL_TYPES,
        "cell2location_genes": len(regression_genes),
        "destvi_genes": len(destvi_genes),
        "gpu": torch.cuda.get_device_name(0),
        "versions": {
            "torch": torch.__version__,
            "scvi_tools": scvi.__version__,
        },
    }
    (RESULT_DIR / "deconvolution_run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved {FINAL_PATH}", flush=True)


if __name__ == "__main__":
    main()
