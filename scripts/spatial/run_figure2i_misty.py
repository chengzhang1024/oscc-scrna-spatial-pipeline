"""Run patient-aware LIANA+ MISTy for the six Stage IVA OSCC patients."""

from __future__ import annotations

import json
from importlib.metadata import version
import warnings

import anndata as ad
import numpy as np
import pandas as pd

import run_figure2c_misty as base


OUTPUT_DIR = base.ROOT / "results" / "spatial_misty" / "figure2i_advanced"
ADVANCED_STAGE = "Stage IVA"


def patient_aggregate_interactions(interactions: pd.DataFrame) -> pd.DataFrame:
    return (
        interactions.groupby(
            ["patient_id", "view", "target", "predictor"], observed=True
        )
        .agg(
            importance_z=("importance_z", "median"),
            mean_importance_z_across_sections=("importance_z", "mean"),
            n_sections=("sample_id", "nunique"),
        )
        .reset_index()
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    adata = ad.read_h5ad(base.INPUT, backed="r")
    proportions = pd.DataFrame(
        np.asarray(adata.obsm["cell2location_proportions"]),
        index=adata.obs_names,
        columns=base.CELL_TYPES,
    )
    metadata = (
        adata.obs[["sample_id", "patient_id", "stage"]]
        .assign(
            sample_id=lambda frame: frame["sample_id"].astype(str),
            patient_id=lambda frame: frame["patient_id"].astype(str),
            stage=lambda frame: frame["stage"].astype(str),
        )
        .drop_duplicates("sample_id")
    )
    metadata = metadata.loc[metadata["stage"].eq(ADVANCED_STAGE)].sort_values("sample_id")
    sample_ids = metadata["sample_id"].tolist()

    all_metrics: list[pd.DataFrame] = []
    all_interactions: list[pd.DataFrame] = []
    warnings.filterwarnings("ignore", category=FutureWarning)
    for sample_id in sample_ids:
        row = metadata.loc[metadata["sample_id"].eq(sample_id)].iloc[0]
        composition = base.make_composition_adata(adata, proportions, sample_id)
        print(
            f"Running {sample_id} | {row['patient_id']} | {row['stage']} "
            f"| {composition.n_obs:,} spots",
            flush=True,
        )
        misty = base.genericMistyData(
            intra=composition,
            extra=None,
            nz_threshold=0,
            bandwidth=base.PARA_BANDWIDTH,
            zoi=base.PARA_ZOI,
            cutoff=base.KERNEL_CUTOFF,
            n_neighs=base.N_NEIGHS,
            max_neighs=base.PARA_MAX_NEIGHS,
            set_diag=False,
            verbose=False,
        )
        metrics, interactions = misty(
            model=base.RandomForestModel,
            seed=base.SEED,
            n_estimators=base.N_ESTIMATORS,
            n_jobs=-1,
            min_samples_leaf=5,
            max_features="sqrt",
            predict_self=False,
            k_cv=10,
            inplace=False,
            verbose=False,
        )
        for frame in (metrics, interactions):
            frame.insert(0, "sample_id", sample_id)
            frame.insert(1, "patient_id", str(row["patient_id"]))
            frame.insert(2, "stage", str(row["stage"]))
        all_metrics.append(metrics)
        all_interactions.append(interactions)

    metrics = pd.concat(all_metrics, ignore_index=True)
    interactions = base.standardize_importance(
        pd.concat(all_interactions, ignore_index=True)
    )
    patient_interactions = patient_aggregate_interactions(interactions)
    group_summary = (
        patient_interactions.groupby(["view", "target", "predictor"], observed=True)
        .agg(
            median_importance_z=("importance_z", "median"),
            mean_importance_z=("importance_z", "mean"),
            stability_fraction=("importance_z", lambda values: float(np.mean(values >= 1.0))),
            n_patients=("patient_id", "nunique"),
        )
        .reset_index()
    )

    metric_columns = [
        "intra_R2", "multi_R2", "gain_R2", "intra", "juxta", "para"
    ]
    patient_metrics = (
        metrics.groupby(["patient_id", "target"], observed=True)[metric_columns]
        .median()
        .reset_index()
    )
    metric_summary = (
        patient_metrics.groupby("target", observed=True)
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

    metrics.to_csv(OUTPUT_DIR / "advanced_misty_target_metrics_by_sample.csv", index=False)
    interactions.to_csv(OUTPUT_DIR / "advanced_misty_interactions_by_sample.csv", index=False)
    patient_metrics.to_csv(OUTPUT_DIR / "advanced_misty_target_metrics_by_patient.csv", index=False)
    patient_interactions.to_csv(
        OUTPUT_DIR / "advanced_misty_interactions_by_patient.csv", index=False
    )
    group_summary.to_csv(OUTPUT_DIR / "advanced_misty_group_importance.csv", index=False)
    metric_summary.to_csv(OUTPUT_DIR / "advanced_misty_group_metrics.csv", index=False)

    run_metadata = {
        "input": str(base.INPUT.relative_to(base.ROOT)),
        "input_view": "cell2location_proportions",
        "group": "Advanced OSCC (Stage IVA)",
        "samples": sample_ids,
        "patients": sorted(metadata["patient_id"].unique().tolist()),
        "cell_types": base.CELL_TYPES,
        "software": {"liana": version("liana"), "anndata": version("anndata")},
        "model": "LIANA+ MISTy RandomForestModel",
        "seed": base.SEED,
        "random_forest": {
            "n_estimators": base.N_ESTIMATORS,
            "min_samples_leaf": 5,
            "max_features": "sqrt",
            "n_jobs": -1,
        },
        "views": {
            "intra": "same Visium spot",
            "juxta_5": {"n_neighs": base.N_NEIGHS},
            "para_15": {
                "coordinate_system": "Visium array_col/array_row",
                "bandwidth": base.PARA_BANDWIDTH,
                "zone_of_indifference": base.PARA_ZOI,
                "max_neighs": base.PARA_MAX_NEIGHS,
                "kernel_cutoff": base.KERNEL_CUTOFF,
            },
        },
        "aggregation": (
            "fit per section; standardize within section/view/target; median across "
            "sections within patient; median across six patients"
        ),
        "stability_definition": (
            "fraction of patients with patient-level importance z-score >= 1; "
            "stable requires >=5/6 patients"
        ),
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2), encoding="utf-8"
    )
    print(f"Saved results to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
