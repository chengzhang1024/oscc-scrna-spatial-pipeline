"""Project Stage IVA spots into the fixed early OSCC niche model and plot Figure 2L."""

from __future__ import annotations

import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
from sklearn.cluster import KMeans

import plot_figure2f_reference_style as base


OUTPUT = base.OUTPUT_DIR / "Fig2L_spatial_niches_advanced_OSCC"
ASSIGNMENT_AUDIT = base.OUTPUT_DIR / "Fig2L_niche_assignments_advanced_OSCC.csv.gz"
ENRICHMENT_AUDIT = base.OUTPUT_DIR / "Fig2L_niche_enrichment_advanced_OSCC.csv"
PATIENT_ENRICHMENT_AUDIT = base.OUTPUT_DIR / "Fig2L_niche_enrichment_by_patient.csv"
PREVALENCE_AUDIT = base.OUTPUT_DIR / "Fig2L_niche_patient_prevalence.csv"
STAGE_COMPARISON_AUDIT = base.OUTPUT_DIR / "Fig2L_niche_stage_comparison.csv"
METADATA_AUDIT = base.OUTPUT_DIR / "Fig2L_niche_projection_metadata.json"
EARLY_ASSIGNMENT = base.OUTPUT_DIR / "Fig2F_niche_assignments_early_OSCC.csv.gz"
SELECTION = base.OUTPUT_DIR / "Fig2G_representative_sample_selection.csv"

N_CLUSTERS = 3
OOD_QUANTILE = 0.99


def bh_adjust(pvalues: list[float]) -> np.ndarray:
    values = np.asarray(pvalues, dtype=float)
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    output = np.empty_like(adjusted)
    output[order] = np.clip(adjusted, 0, 1)
    return output


def patient_fractions(
    patient_ids: np.ndarray, labels: np.ndarray, cluster_to_id: dict[int, str]
) -> pd.DataFrame:
    rows = []
    for patient_id in sorted(set(patient_ids)):
        selected = patient_ids == patient_id
        for cluster, niche_id in cluster_to_id.items():
            rows.append(
                {
                    "patient_id": patient_id,
                    "niche_id": niche_id,
                    "fraction": float(np.mean(labels[selected] == cluster)),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    selection = pd.read_csv(SELECTION)
    representative = selection.loc[
        selection["selected"].astype(bool), "sample_id"
    ].astype(str).tolist()
    if len(representative) != 1:
        raise ValueError("Figure 2G selection audit must contain one representative sample")
    representative_sample = representative[0]

    adata = ad.read_h5ad(base.INPUT)
    proportions_all = np.asarray(adata.obsm["cell2location_proportions"], dtype=float)
    coordinates_all = np.asarray(adata.obsm["spatial"], dtype=float)
    stages = adata.obs["stage"].astype(str)
    early_mask = stages.isin(["Stage I", "Stage II"]).to_numpy()
    advanced_mask = stages.eq("Stage IVA").to_numpy()

    early_obs = adata.obs.loc[early_mask, ["sample_id", "patient_id", "stage"]].copy()
    advanced_obs = adata.obs.loc[
        advanced_mask, ["sample_id", "patient_id", "stage"]
    ].copy()
    for frame in (early_obs, advanced_obs):
        frame["sample_id"] = frame["sample_id"].astype(str)
        frame["patient_id"] = frame["patient_id"].astype(str)

    early_local, _ = base.build_local_profiles(
        proportions_all[early_mask],
        coordinates_all[early_mask],
        early_obs["sample_id"].to_numpy(),
    )
    advanced_local, _ = base.build_local_profiles(
        proportions_all[advanced_mask],
        coordinates_all[advanced_mask],
        advanced_obs["sample_id"].to_numpy(),
    )
    early_features = base.patient_centered_clr(
        early_local, early_obs["patient_id"].to_numpy()
    )
    advanced_features = base.patient_centered_clr(
        advanced_local, advanced_obs["patient_id"].to_numpy()
    )
    train_index = base.balanced_training_index(early_obs["patient_id"].to_numpy())
    model = KMeans(n_clusters=N_CLUSTERS, n_init=50, random_state=base.SEED)
    model.fit(early_features[train_index])
    early_labels = model.predict(early_features)
    advanced_labels = model.predict(advanced_features)

    previous = pd.read_csv(EARLY_ASSIGNMENT).set_index("spot_id").loc[early_obs.index]
    agreement = float(np.mean(early_labels == previous["cluster"].to_numpy(dtype=int)))
    if agreement < 0.999:
        raise RuntimeError(
            f"Refitted early K=3 model reproduced only {agreement:.3%} of assignments"
        )
    niche_lookup = previous.drop_duplicates("cluster").set_index("cluster")
    cluster_to_id = niche_lookup["niche_id"].astype(str).to_dict()
    cluster_to_name = niche_lookup["niche_name"].astype(str).to_dict()
    cluster_to_color = niche_lookup["niche_color"].astype(str).to_dict()

    early_distances = np.linalg.norm(
        early_features - model.cluster_centers_[early_labels], axis=1
    )
    thresholds = {
        cluster: float(np.quantile(early_distances[early_labels == cluster], OOD_QUANTILE))
        for cluster in range(N_CLUSTERS)
    }
    advanced_distances = np.linalg.norm(
        advanced_features - model.cluster_centers_[advanced_labels], axis=1
    )
    advanced_thresholds = np.asarray([thresholds[int(label)] for label in advanced_labels])
    ood = advanced_distances > advanced_thresholds

    # OOD spots remain auditable nearest-niche projections but are excluded from
    # the niche enrichment matrix and shown in gray on the representative section.
    enrichment_labels = advanced_labels.copy()
    enrichment_labels[ood] = -1
    base.MIN_PATIENTS_PRESENT = 5
    matrix, enrichment, by_patient = base.enrichment_tables(
        advanced_local,
        enrichment_labels,
        advanced_obs["patient_id"].to_numpy(),
        N_CLUSTERS,
    )

    base.OUTPUT = OUTPUT
    base.PANEL_LETTER = "L"
    base.SPATIAL_TITLE = "Niche-advanced OSCC"
    base.HEATMAP_TITLE = "Abundance-advanced OSCC by niche"
    base.plot_figure(
        adata,
        advanced_mask,
        advanced_obs,
        coordinates_all[advanced_mask],
        advanced_labels,
        matrix,
        enrichment,
        cluster_to_name,
        representative_sample,
        fixed_cluster_to_id=cluster_to_id,
        fixed_cluster_to_color=cluster_to_color,
        ood_mask=ood,
    )

    enrichment["niche_id"] = enrichment["cluster"].map(cluster_to_id)
    enrichment["niche_name"] = enrichment["cluster"].map(cluster_to_name)
    enrichment["niche_color"] = enrichment["cluster"].map(cluster_to_color)
    by_patient["niche_id"] = by_patient["cluster"].map(cluster_to_id)
    by_patient["niche_name"] = by_patient["cluster"].map(cluster_to_name)

    assignment = advanced_obs.reset_index(names="spot_id")
    assignment["x"] = coordinates_all[advanced_mask, 0]
    assignment["y"] = coordinates_all[advanced_mask, 1]
    assignment["projected_cluster"] = advanced_labels
    assignment["projected_niche_id"] = assignment["projected_cluster"].map(cluster_to_id)
    assignment["projected_niche_name"] = assignment["projected_cluster"].map(cluster_to_name)
    assignment["projected_niche_color"] = assignment["projected_cluster"].map(cluster_to_color)
    assignment["distance_to_early_centroid"] = advanced_distances
    assignment["early_cluster_q99_distance"] = advanced_thresholds
    assignment["out_of_distribution"] = ood
    assignment["display_niche_id"] = np.where(
        ood, "OOD", assignment["projected_niche_id"]
    )

    nearest_prevalence = patient_fractions(
        advanced_obs["patient_id"].to_numpy(), advanced_labels, cluster_to_id
    )
    nearest_prevalence["stage_group"] = "Advanced Stage IVA"
    nearest_prevalence["fraction_definition"] = "nearest early niche; includes OOD"
    display_rows = []
    for patient_id in sorted(advanced_obs["patient_id"].unique()):
        selected = advanced_obs["patient_id"].to_numpy() == patient_id
        for cluster, niche_id in cluster_to_id.items():
            display_rows.append(
                {
                    "patient_id": patient_id,
                    "niche_id": niche_id,
                    "fraction": float(np.mean((advanced_labels == cluster)[selected] & ~ood[selected])),
                    "stage_group": "Advanced Stage IVA",
                    "fraction_definition": "in-distribution display fraction",
                }
            )
        display_rows.append(
            {
                "patient_id": patient_id,
                "niche_id": "OOD",
                "fraction": float(np.mean(ood[selected])),
                "stage_group": "Advanced Stage IVA",
                "fraction_definition": "out-of-distribution fraction",
            }
        )
    prevalence = pd.concat([nearest_prevalence, pd.DataFrame(display_rows)], ignore_index=True)

    early_patient_fraction = patient_fractions(
        early_obs["patient_id"].to_numpy(), early_labels, cluster_to_id
    )
    comparison_rows = []
    for cluster, niche_id in sorted(cluster_to_id.items(), key=lambda item: item[1]):
        early_values = early_patient_fraction.loc[
            early_patient_fraction["niche_id"].eq(niche_id), "fraction"
        ].to_numpy()
        advanced_values = nearest_prevalence.loc[
            nearest_prevalence["niche_id"].eq(niche_id), "fraction"
        ].to_numpy()
        test = mannwhitneyu(early_values, advanced_values, alternative="two-sided", method="exact")
        comparison_rows.append(
            {
                "niche_id": niche_id,
                "niche_name": cluster_to_name[cluster],
                "early_n_patients": len(early_values),
                "advanced_n_patients": len(advanced_values),
                "early_median_fraction": float(np.median(early_values)),
                "advanced_median_fraction": float(np.median(advanced_values)),
                "advanced_minus_early_median": float(np.median(advanced_values) - np.median(early_values)),
                "mannwhitney_u": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )
    comparison = pd.DataFrame(comparison_rows)
    comparison["p_adj_bh"] = bh_adjust(comparison["p_value"].tolist())

    ASSIGNMENT_AUDIT.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_csv(ASSIGNMENT_AUDIT, index=False, compression="gzip")
    enrichment.sort_values(["niche_id", "cell_type"]).to_csv(ENRICHMENT_AUDIT, index=False)
    by_patient.sort_values(["patient_id", "niche_id", "cell_type"]).to_csv(
        PATIENT_ENRICHMENT_AUDIT, index=False
    )
    prevalence.sort_values(["fraction_definition", "patient_id", "niche_id"]).to_csv(
        PREVALENCE_AUDIT, index=False
    )
    comparison.to_csv(STAGE_COMPARISON_AUDIT, index=False)
    metadata = {
        "input": str(base.INPUT.relative_to(base.ROOT)),
        "group": "Advanced OSCC (Stage IVA)",
        "representative_sample": representative_sample,
        "n_spots": int(advanced_mask.sum()),
        "samples": sorted(advanced_obs["sample_id"].unique().tolist()),
        "patients": sorted(advanced_obs["patient_id"].unique().tolist()),
        "projection_model": "fixed Figure 2F early KMeans K=3 model",
        "early_model_reproduction_fraction": agreement,
        "input_view": "cell2location_proportions",
        "local_profile": "center spot plus five nearest spots, arithmetic mean",
        "transform": "CLR followed by within-patient median centering",
        "training": f"up to {base.TRAIN_PER_PATIENT} early spots per patient with equal patient weight",
        "seed": base.SEED,
        "niche_mapping": {
            cluster_to_id[cluster]: {
                "cluster": cluster,
                "name": cluster_to_name[cluster],
                "color": cluster_to_color[cluster],
            }
            for cluster in sorted(cluster_to_id)
        },
        "ood_definition": (
            "distance to assigned early centroid exceeds the cluster-specific 99th "
            "percentile among early spots"
        ),
        "ood_fraction": float(np.mean(ood)),
        "ood_fraction_representative": float(
            np.mean(ood[advanced_obs["sample_id"].eq(representative_sample).to_numpy()])
        ),
        "heatmap": (
            "patient-level log2 observed/expected among in-distribution projected spots; "
            "group median across six patients"
        ),
        "dot_definition": (
            "same enrichment direction in >=5/6 patients and absolute median "
            f"log2(Ro/e) >= {base.DOT_MIN_ABS_LOG2_ROE}"
        ),
        "stage_test": "two-sided exact Mann-Whitney U on patient-level nearest-niche fractions; BH across three niches",
        "interpretation": (
            "Projection tests preservation or redistribution of early composition-defined "
            "niches; it does not define new advanced-specific functional niches"
        ),
    }
    METADATA_AUDIT.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(
        f"Projected {len(advanced_obs):,} Stage IVA spots; OOD={np.mean(ood):.1%}; "
        f"representative {representative_sample} OOD={metadata['ood_fraction_representative']:.1%}"
    )
    print(f"Saved {OUTPUT.with_suffix('.png')}")
    print(f"Saved {OUTPUT.with_suffix('.pdf')}")
    print(f"Saved audits under {base.OUTPUT_DIR}")


if __name__ == "__main__":
    main()
