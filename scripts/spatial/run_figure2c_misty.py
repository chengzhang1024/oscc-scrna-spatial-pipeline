"""Run LIANA+ MISTy for the four Early-stage GSE208253 sections."""

from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path
import warnings

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))
os.environ.setdefault("MPLBACKEND", "Agg")

import anndata as ad
from anndata import AnnData
import numpy as np
import pandas as pd

from liana.method import genericMistyData
from liana.method.sp._misty._single_view_models import RandomForestModel


INPUT = ROOT / "data" / "processed" / "GSE208253" / "GSE208253_cleaned_deconvolved.h5ad"
OUTPUT_DIR = ROOT / "results" / "spatial_misty" / "figure2c_early"

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

EARLY_STAGES = {"Stage I", "Stage II"}
SEED = 181919
N_NEIGHS = 5
PARA_BANDWIDTH = 15.0
PARA_ZOI = 2.1
PARA_MAX_NEIGHS = 60
KERNEL_CUTOFF = 0.05
N_ESTIMATORS = 200


def make_composition_adata(
    adata: ad.AnnData, proportions: pd.DataFrame, sample_id: str
) -> AnnData:
    selected = adata.obs["sample_id"].astype(str).eq(sample_id)
    obs = adata.obs.loc[selected].copy()
    sample = AnnData(
        X=proportions.loc[selected, CELL_TYPES].to_numpy(dtype=np.float32),
        obs=obs,
        var=pd.DataFrame(index=CELL_TYPES),
    )
    # Array coordinates make the published juxta_5 and para_15 parameters explicit.
    sample.obsm["spatial"] = obs[["array_col", "array_row"]].to_numpy(dtype=float)
    return sample


def standardize_importance(interactions: pd.DataFrame) -> pd.DataFrame:
    output = interactions.copy()

    def zscore(values: pd.Series) -> pd.Series:
        standard_deviation = float(values.std(ddof=0))
        if standard_deviation == 0:
            return pd.Series(np.zeros(len(values)), index=values.index)
        return (values - float(values.mean())) / standard_deviation

    output["importance_z"] = output.groupby(
        ["sample_id", "view", "target"], observed=True
    )["importances"].transform(zscore)
    return output


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(INPUT, backed="r")
    proportions = pd.DataFrame(
        np.asarray(adata.obsm["cell2location_proportions"]),
        index=adata.obs_names,
        columns=CELL_TYPES,
    )

    metadata = (
        adata.obs[["sample_id", "patient_id", "stage"]]
        .assign(
            sample_id=lambda frame: frame["sample_id"].astype(str),
            stage=lambda frame: frame["stage"].astype(str),
        )
        .drop_duplicates("sample_id")
    )
    metadata = metadata.loc[metadata["stage"].isin(EARLY_STAGES)].sort_values("sample_id")
    sample_ids = metadata["sample_id"].tolist()

    all_metrics: list[pd.DataFrame] = []
    all_interactions: list[pd.DataFrame] = []
    warnings.filterwarnings("ignore", category=FutureWarning)

    for sample_id in sample_ids:
        row = metadata.loc[metadata["sample_id"].eq(sample_id)].iloc[0]
        composition = make_composition_adata(adata, proportions, sample_id)
        print(
            f"Running {sample_id} | {row['patient_id']} | {row['stage']} "
            f"| {composition.n_obs:,} spots",
            flush=True,
        )
        misty = genericMistyData(
            intra=composition,
            extra=None,
            nz_threshold=0,
            bandwidth=PARA_BANDWIDTH,
            zoi=PARA_ZOI,
            cutoff=KERNEL_CUTOFF,
            n_neighs=N_NEIGHS,
            max_neighs=PARA_MAX_NEIGHS,
            set_diag=False,
            verbose=False,
        )
        metrics, interactions = misty(
            model=RandomForestModel,
            seed=SEED,
            n_estimators=N_ESTIMATORS,
            n_jobs=-1,
            min_samples_leaf=5,
            max_features="sqrt",
            predict_self=False,
            k_cv=10,
            inplace=False,
            verbose=False,
        )
        metrics.insert(0, "sample_id", sample_id)
        metrics.insert(1, "patient_id", str(row["patient_id"]))
        metrics.insert(2, "stage", str(row["stage"]))
        interactions.insert(0, "sample_id", sample_id)
        interactions.insert(1, "patient_id", str(row["patient_id"]))
        interactions.insert(2, "stage", str(row["stage"]))
        all_metrics.append(metrics)
        all_interactions.append(interactions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    interactions = standardize_importance(pd.concat(all_interactions, ignore_index=True))
    group_summary = (
        interactions.groupby(["view", "target", "predictor"], observed=True)
        .agg(
            median_importance_z=("importance_z", "median"),
            mean_importance_z=("importance_z", "mean"),
            stability_fraction=("importance_z", lambda values: float(np.mean(values >= 1.0))),
            n_patients=("patient_id", "nunique"),
        )
        .reset_index()
    )
    metric_summary = (
        metrics.groupby("target", observed=True)
        .agg(
            median_intra_R2=("intra_R2", "median"),
            median_multi_R2=("multi_R2", "median"),
            median_gain_R2=("gain_R2", "median"),
            median_intra_contribution=("intra", "median"),
            median_juxta_contribution=("juxta", "median"),
            median_para_contribution=("para", "median"),
            n_patients=("patient_id", "nunique"),
        )
        .reset_index()
    )

    metrics.to_csv(OUTPUT_DIR / "early_misty_target_metrics_by_sample.csv", index=False)
    interactions.to_csv(OUTPUT_DIR / "early_misty_interactions_by_sample.csv", index=False)
    group_summary.to_csv(OUTPUT_DIR / "early_misty_group_importance.csv", index=False)
    metric_summary.to_csv(OUTPUT_DIR / "early_misty_group_metrics.csv", index=False)

    run_metadata = {
        "input": str(INPUT.relative_to(ROOT)),
        "input_view": "cell2location_proportions",
        "group": "Early-stage OSCC (Stage I-II)",
        "samples": sample_ids,
        "patients": metadata["patient_id"].astype(str).tolist(),
        "cell_types": CELL_TYPES,
        "software": {"liana": version("liana"), "anndata": version("anndata")},
        "model": "LIANA+ MISTy RandomForestModel",
        "seed": SEED,
        "random_forest": {
            "n_estimators": N_ESTIMATORS,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "n_jobs": -1,
        },
        "views": {
            "intra": "same Visium spot",
            "juxta_5": {"n_neighs": N_NEIGHS},
            "para_15": {
                "coordinate_system": "Visium array_col/array_row",
                "bandwidth": PARA_BANDWIDTH,
                "zone_of_indifference": PARA_ZOI,
                "max_neighs": PARA_MAX_NEIGHS,
                "kernel_cutoff": KERNEL_CUTOFF,
            },
        },
        "aggregation": "patient-level median standardized feature importance",
        "stability_definition": "fraction of patients with within-target importance z-score >= 1",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved results to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
