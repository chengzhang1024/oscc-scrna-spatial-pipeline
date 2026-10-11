"""Reproduce Supplementary Figure S1 and its numerical source tables.

This script reads existing source outputs without modifying them. All new analyses
are explicitly exploratory sensitivity checks; there is no outcome-based selection.
Run from any directory: python run_robustness_revision.py --root /path/to/repository
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from matplotlib.colors import TwoSlopeNorm
from scipy.stats import rankdata, spearmanr, wilcoxon

SEED = 20260911
N_BOOTSTRAP = 10000
CLUSTERS = [f"C{i}" for i in range(9)]
MODULES = [f"M{i}" for i in range(1, 9)]
METHODS = ["cell_weighted", "patient_balanced", "complexity_adjusted"]
METHOD_LABELS = ["Cell\nweighted", "Patient\nbalanced", "Complexity\nadjusted"]


def bh(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    result = np.empty(len(p))
    result[order] = np.minimum(adjusted, 1.0)
    return result


def clean_value(value):
    if isinstance(value, dict):
        return {str(k): clean_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_value(v) for v in value]
    if isinstance(value, np.generic):
        return clean_value(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[3])
    args = parser.parse_args()
    root = args.root.resolve()
    out = Path(__file__).resolve().parent
    f3 = root / "results/paper9/figure3_epithelial"
    f4 = root / "results/paper9/figure4_epithelial_hdwgcna"
    sources = {
        "sr_cells": f3 / "Fig3G_epithelial_cells_SR_diffusion.csv.gz",
        "prior_sr_summary": f3 / "Fig3H_epithelial_mean_SR_and_sensitivity.csv",
        "prior_patient_sr": f3 / "Fig3H_epithelial_patient_cluster_SR.csv",
        "primary_gene_modules": f4 / "gene_modules.csv",
        "sensitivity_gene_modules": f4 / "sensitivity_without_C5_C8_gene_modules.csv",
        "prior_module_matches": f4 / "QC_sensitivity_best_module_matches.csv",
        "sample_module_scores": f4 / "sample_cluster_module_eigengenes.csv",
        "module_cell_metadata": f4 / "input/cells.csv",
        "cell_module_scores": f4 / "cell_module_eigengenes.csv",
        "prior_paired_associations": f4 / "patient_paired_module_associations.csv",
        "sr_analysis_script": root / "36_plot_figure3h_epithelial_mean_sr_reference_style.py",
        "module_analysis_script": root / "scripts/42_plot_figure4_hdwgcna.py",
        "module_sensitivity_script": root / "scripts/43_check_figure4_qc_sensitivity.R",
    }
    cells = pd.read_csv(sources["sr_cells"])
    original_sr = pd.read_csv(sources["prior_sr_summary"]).set_index("epithelial_cluster").reindex(CLUSTERS)
    assert len(cells) == cells.cell_id.nunique() == 5609
    assert set(cells.epithelial_cluster) == set(CLUSTERS)
    assert cells.groupby("sample_id").patient_id.nunique().max() == 1
    assert not cells[["patient_id", "sample_id", "signaling_entropy_rate", "n_genes_by_counts", "total_counts"]].isna().any().any()
    patients = sorted(cells.patient_id.unique(), key=lambda x: int(x[1:]))
    patient_codes = pd.Categorical(cells.patient_id, categories=patients).codes
    cluster_codes = pd.Categorical(cells.epithelial_cluster, categories=CLUSTERS).codes
    sr = cells.signaling_entropy_rate.to_numpy(float)
    qc = np.column_stack([np.log1p(cells.n_genes_by_counts), np.log1p(cells.total_counts)])
    qc_mean, qc_sd = qc.mean(axis=0), qc.std(axis=0)
    design = np.column_stack([np.ones(len(cells)), (qc - qc_mean) / qc_sd])
    beta = np.linalg.lstsq(design, sr, rcond=None)[0]
    cells["complexity_adjusted_SR_residual"] = sr - design @ beta
    r_squared = 1 - np.sum(cells.complexity_adjusted_SR_residual.to_numpy() ** 2) / np.sum((sr - sr.mean()) ** 2)

    # patient_id is the biological unit; multiple samples of one patient are pooled
    # within each cluster, exactly as in the pre-existing Figure 3H analysis.
    patient_cluster = cells.groupby(["patient_id", "epithelial_cluster"]).agg(
        n_cells=("cell_id", "size"), n_samples=("sample_id", "nunique"),
        patient_mean_SR=("signaling_entropy_rate", "mean"),
        patient_mean_adjusted_SR_residual=("complexity_adjusted_SR_residual", "mean")
    ).reset_index()
    prior_patient = pd.read_csv(sources["prior_patient_sr"])
    pair_check = patient_cluster.merge(prior_patient, on=["patient_id", "epithelial_cluster"], suffixes=("", "_prior"), validate="one_to_one")
    assert len(pair_check) == len(patient_cluster) == len(prior_patient)
    assert np.allclose(pair_check.patient_mean_SR, pair_check.patient_mean_SR_prior, atol=1e-12, rtol=0)
    summary = cells.groupby("epithelial_cluster").agg(
        n_cells=("cell_id", "size"), n_patients=("patient_id", "nunique"),
        n_samples=("sample_id", "nunique"), cell_weighted_mean_SR=("signaling_entropy_rate", "mean"),
        complexity_adjusted_mean_SR_residual=("complexity_adjusted_SR_residual", "mean")
    ).reindex(CLUSTERS)
    summary["patient_balanced_mean_SR"] = patient_cluster.groupby("epithelial_cluster").patient_mean_SR.mean()
    summary["largest_patient_cell_fraction"] = patient_cluster.groupby("epithelial_cluster").n_cells.max() / summary.n_cells
    summary["min_cells_per_patient"] = patient_cluster.groupby("epithelial_cluster").n_cells.min()
    summary["max_cells_per_patient"] = patient_cluster.groupby("epithelial_cluster").n_cells.max()
    summary["qc_flag"] = ["mixed_mesenchymal_muscle_like" if c == "C5" else "immune_like" if c == "C8" else "none" for c in CLUSTERS]
    for method, column in zip(METHODS, ["cell_weighted_mean_SR", "patient_balanced_mean_SR", "complexity_adjusted_mean_SR_residual"]):
        summary[method + "_rank"] = summary[column].rank(ascending=False, method="min").astype(int)
    source_checks = {}
    for new, old in [("cell_weighted_mean_SR", "mean_SR"), ("patient_balanced_mean_SR", "equal_patient_weight_mean_SR"), ("complexity_adjusted_mean_SR_residual", "mean_adjusted_SR_residual")]:
        source_checks[new + "_max_abs_difference"] = float(np.max(np.abs(summary[new] - original_sr[old])))
        assert source_checks[new + "_max_abs_difference"] < 1e-12

    # Sufficient statistics permit a true whole-patient bootstrap while refitting
    # the complexity model in each replicate. Global predictor standardization
    # is retained because with an intercept it does not change fitted values.
    npat, nclu = len(patients), len(CLUSTERS)
    p_n = np.zeros((npat, nclu))
    p_sr = np.zeros((npat, nclu))
    p_x = np.zeros((npat, nclu, 3))
    p_xtx = np.zeros((npat, 3, 3))
    p_xty = np.zeros((npat, 3))
    for pi in range(npat):
        mask = patient_codes == pi
        p_xtx[pi] = design[mask].T @ design[mask]
        p_xty[pi] = design[mask].T @ sr[mask]
        for ci in range(nclu):
            selected = mask & (cluster_codes == ci)
            p_n[pi, ci] = selected.sum()
            p_sr[pi, ci] = sr[selected].sum()
            p_x[pi, ci] = design[selected].sum(axis=0)
    p_present = (p_n > 0).astype(float)
    p_mean = np.divide(p_sr, p_n, out=np.zeros_like(p_sr), where=p_n > 0)

    def estimates(weights):
        n = weights @ p_n
        sums = weights @ p_sr
        observed = weights @ p_present
        raw = np.divide(sums, n, out=np.full_like(n, np.nan), where=n > 0)
        balanced = np.divide(weights @ p_mean, observed, out=np.full_like(n, np.nan), where=observed > 0)
        weighted_xtx = np.einsum("bp,pij->bij", weights, p_xtx)
        weighted_xty = weights @ p_xty
        coefficients = np.linalg.solve(weighted_xtx, weighted_xty[..., None])[..., 0]
        weighted_x = np.einsum("bp,pcj->bcj", weights, p_x)
        residual_sums = sums - np.einsum("bcj,bj->bc", weighted_x, coefficients)
        residual = np.divide(residual_sums, n, out=np.full_like(n, np.nan), where=n > 0)
        return np.stack([raw, balanced, residual], axis=1), n, observed

    original_estimates, _, _ = estimates(np.ones((1, npat)))
    assert np.allclose(original_estimates[0, 0], summary.cell_weighted_mean_SR, atol=1e-12, rtol=0)
    assert np.allclose(original_estimates[0, 1], summary.patient_balanced_mean_SR, atol=1e-12, rtol=0)
    assert np.allclose(original_estimates[0, 2], summary.complexity_adjusted_mean_SR_residual, atol=1e-12, rtol=0)
    rng = np.random.default_rng(SEED)
    weights = rng.multinomial(npat, np.repeat(1 / npat, npat), size=N_BOOTSTRAP)
    boot_estimates, boot_ncells, boot_npatients = estimates(weights)
    # Check the sufficient-statistic implementation against literal whole-patient
    # cell replication, including duplication of a patient's cells and model refit.
    direct_bootstrap_errors = []
    for replicate in range(5):
        indices = np.repeat(np.arange(len(cells)), weights[replicate, patient_codes])
        x_direct, y_direct = design[indices], sr[indices]
        beta_direct = np.linalg.lstsq(x_direct, y_direct, rcond=None)[0]
        direct = pd.DataFrame({"cluster": cells.epithelial_cluster.iloc[indices].to_numpy(),
            "raw": y_direct, "residual": y_direct - x_direct @ beta_direct})
        direct_cluster = direct.groupby("cluster")[["raw", "residual"]].mean().reindex(CLUSTERS)
        direct_balanced = []
        for ci in range(nclu):
            duplicated_means = np.repeat(p_mean[:, ci][p_present[:, ci] > 0], weights[replicate][p_present[:, ci] > 0])
            direct_balanced.append(duplicated_means.mean() if len(duplicated_means) else np.nan)
        direct_estimates = np.stack([direct_cluster.raw.to_numpy(), direct_balanced, direct_cluster.residual.to_numpy()])
        error = float(np.nanmax(np.abs(direct_estimates - boot_estimates[replicate])))
        direct_bootstrap_errors.append(error)
        assert np.allclose(direct_estimates, boot_estimates[replicate], equal_nan=True, atol=1e-11, rtol=0)
    complete = np.isfinite(boot_estimates).all(axis=(1, 2))
    boot_ranks = np.full_like(boot_estimates, np.nan)
    boot_ranks[complete] = rankdata(-boot_estimates[complete], axis=2, method="min")
    rows = []
    for mi, method in enumerate(METHODS):
        for ci, cluster in enumerate(CLUSTERS):
            good = np.isfinite(boot_estimates[:, mi, ci])
            values = boot_estimates[good, mi, ci]
            ranks = boot_ranks[complete, mi, ci]
            rows.append(dict(method=method, epithelial_cluster=cluster,
                estimate=original_estimates[0, mi, ci],
                bootstrap_ci95_low=np.quantile(values, 0.025), bootstrap_ci95_high=np.quantile(values, 0.975),
                evaluable_estimate_replicates=int(good.sum()),
                complete_cluster_rank_replicates=int(complete.sum()),
                rank1_fraction_complete_replicates=float(np.mean(ranks == 1)),
                median_rank_complete_replicates=float(np.median(ranks)),
                rank_ci95_low=float(np.quantile(ranks, 0.025)), rank_ci95_high=float(np.quantile(ranks, 0.975))))
    boot_summary = pd.DataFrame(rows)
    for tail in ["low", "high"]:
        summary["patient_bootstrap_ci95_" + tail] = boot_summary[boot_summary.method.eq("patient_balanced")].set_index("epithelial_cluster")["bootstrap_ci95_" + tail]
    replicate_rows = []
    for mi, method in enumerate(METHODS):
        for ci, cluster in enumerate(CLUSTERS):
            replicate_rows.append(pd.DataFrame(dict(replicate=np.arange(1, N_BOOTSTRAP + 1), method=method,
                epithelial_cluster=cluster, estimate=boot_estimates[:, mi, ci],
                all_nine_clusters_observed=complete, rank_if_all_clusters_observed=boot_ranks[:, mi, ci],
                sampled_patient_copies_with_cluster=boot_npatients[:, ci], sampled_cells=boot_ncells[:, ci])))
    pd.concat(replicate_rows, ignore_index=True).to_csv(out / "exploratory_patient_bootstrap_replicates.csv.gz", index=False, compression="gzip")
    pd.DataFrame(weights, columns=patients).rename_axis("replicate_zero_based").to_csv(out / "exploratory_patient_bootstrap_weights.csv.gz", compression="gzip")

    loo_weights = np.ones((npat, npat)) - np.eye(npat)
    loo_estimates, loo_ncells, loo_npatients = estimates(loo_weights)
    loo_complete = np.isfinite(loo_estimates).all(axis=(1, 2))
    loo_ranks = np.full_like(loo_estimates, np.nan)
    loo_ranks[loo_complete] = rankdata(-loo_estimates[loo_complete], axis=2, method="min")
    loo_rows = []
    for pi, omitted in enumerate(patients):
        for mi, method in enumerate(METHODS):
            for ci, cluster in enumerate(CLUSTERS):
                loo_rows.append(dict(omitted_patient=omitted, method=method, epithelial_cluster=cluster,
                    estimate=loo_estimates[pi, mi, ci], rank_if_all_clusters_observed=loo_ranks[pi, mi, ci],
                    remaining_cells=int(loo_ncells[pi, ci]), remaining_patients=int(loo_npatients[pi, ci]),
                    all_nine_clusters_observed=bool(loo_complete[pi])))
    loo = pd.DataFrame(loo_rows)

    primary = pd.read_csv(sources["primary_gene_modules"])
    sensitivity = pd.read_csv(sources["sensitivity_gene_modules"])
    assert primary.gene_name.is_unique and sensitivity.gene_name.is_unique
    assert set(primary.gene_name) == set(sensitivity.gene_name)
    overlap_rows = []
    for m in MODULES:
        a = set(primary.loc[primary.module.eq(m), "gene_name"])
        for s in sensitivity.module.unique():
            if s == "grey":
                continue
            b = set(sensitivity.loc[sensitivity.module.eq(s), "gene_name"])
            overlap_rows.append(dict(primary_module=m, sensitivity_module=s, primary_size=len(a), sensitivity_size=len(b),
                overlap=len(a & b), jaccard=len(a & b) / len(a | b), fraction_primary_retained=len(a & b) / len(a)))
    overlaps = pd.DataFrame(overlap_rows)
    best = overlaps.loc[overlaps.groupby("primary_module").jaccard.idxmax()].reset_index(drop=True)
    old_best = pd.read_csv(sources["prior_module_matches"])
    comparison = best.merge(old_best, on=["primary_module", "sensitivity_module"], suffixes=("", "_prior"), validate="one_to_one")
    assert len(comparison) == len(MODULES)
    for column in ["primary_size", "sensitivity_size", "overlap", "jaccard", "fraction_primary_retained"]:
        assert np.allclose(comparison[column], comparison[column + "_prior"], atol=1e-12, rtol=0)

    sample_scores = pd.read_csv(sources["sample_module_scores"])
    module_cells = pd.read_csv(sources["module_cell_metadata"])
    raw_me = pd.read_csv(sources["cell_module_scores"]).set_index("r_cell_id").reindex(module_cells.r_cell_id)[MODULES].reset_index(drop=True)
    assert not raw_me.isna().any().any()
    standardized_me = (raw_me - raw_me.mean()) / raw_me.std(ddof=0)
    joined_me = pd.concat([module_cells[["sample_id", "patient_id", "epithelial_cluster"]], standardized_me], axis=1)
    independently_aggregated_me = joined_me.groupby(["sample_id", "patient_id", "epithelial_cluster"]).agg(
        **{m: (m, "mean") for m in MODULES}, n_cells=(MODULES[0], "size")).reset_index()
    check_me = independently_aggregated_me.merge(sample_scores, on=["sample_id", "patient_id", "epithelial_cluster"], suffixes=("", "_prior"), validate="one_to_one")
    assert len(check_me) == len(sample_scores) == len(independently_aggregated_me)
    module_me_max_error = 0.0
    for column in MODULES + ["n_cells"]:
        module_me_max_error = max(module_me_max_error, float(np.abs(check_me[column] - check_me[column + "_prior"]).max()))
        assert np.allclose(check_me[column], check_me[column + "_prior"], atol=1e-12, rtol=0)
    assert sample_scores.groupby("sample_id").patient_id.nunique().max() == 1
    valid = sample_scores[sample_scores.n_cells >= 10]
    paired_rows, patient_contrasts = [], []
    for cluster in CLUSTERS:
        target = valid[valid.epithelial_cluster.eq(cluster)].set_index("sample_id")
        reference = valid[~valid.epithelial_cluster.isin([cluster, "C5", "C8"])].groupby("sample_id")[MODULES].mean()
        common = target.index.intersection(reference.index)
        differences = target.loc[common, MODULES] - reference.loc[common, MODULES]
        differences["patient_id"] = target.loc[common, "patient_id"]
        per_patient = differences.groupby("patient_id")[MODULES].mean()
        for patient, record in per_patient.iterrows():
            for m in MODULES:
                patient_contrasts.append(dict(patient_id=patient, epithelial_cluster=cluster, module=m,
                    paired_contrast=record[m], n_paired_samples=int((differences.patient_id == patient).sum())))
        for m in MODULES:
            values = per_patient[m].dropna().to_numpy()
            tested = len(values) >= 5 and np.any(values != 0)
            pvalue = wilcoxon(values).pvalue if tested else np.nan
            paired_rows.append(dict(module=m, cluster=cluster, n_patients=len(values),
                n_paired_samples=len(common), paired_effect=values.mean() if len(values) else np.nan,
                pvalue=pvalue, inferential_status="tested" if tested else "not_tested_fewer_than_5_patients"))
    paired = pd.DataFrame(paired_rows)
    paired["fdr"] = np.nan
    tested = paired.pvalue.notna()
    paired.loc[tested, "fdr"] = bh(paired.loc[tested, "pvalue"])
    paired["fdr_family_size"] = int(tested.sum())
    paired["fdr_below_0_05"] = paired.fdr < 0.05
    old_paired = pd.read_csv(sources["prior_paired_associations"])
    pcheck = paired.merge(old_paired, on=["module", "cluster"], suffixes=("", "_prior"), validate="one_to_one")
    for column in ["n_patients", "paired_effect", "pvalue", "fdr"]:
        assert np.allclose(pcheck[column], pcheck[column + "_prior"], equal_nan=True, atol=1e-12, rtol=0)

    coverage = cells.groupby(["patient_id", "sample_id", "epithelial_cluster"]).agg(n_cells=("cell_id", "size")).reset_index()
    tables = {
        "S1_SR_cluster_summary": summary.reset_index(),
        "S2_patient_cluster_SR": patient_cluster,
        "S3_patient_sample_coverage": coverage,
        "S4_bootstrap_summary": boot_summary,
        "S5_leave_one_patient_out": loo,
        "S6_module_best_overlap": best,
        "S7_module_all_overlaps": overlaps,
        "S8_patient_module_tests": paired,
        "S9_patient_module_contrasts": pd.DataFrame(patient_contrasts),
    }
    descriptions = {
        "S1_SR_cluster_summary": "Recomputed raw, patient-balanced and complexity-adjusted SR; 95% percentile intervals from whole-patient bootstrap. n_patients is distinct patient_id, not sample_id.",
        "S2_patient_cluster_SR": "Cells pooled within patient_id and cluster across available samples; each observed patient contributes one mean per cluster.",
        "S3_patient_sample_coverage": "Exact patient/sample/cluster cell counts; a sample maps to one patient. Unequal and sparse coverage limits generalization.",
        "S4_bootstrap_summary": "Exploratory 10,000 whole-patient resamples, seed 20260911. Complexity model refit per resample. Rank frequencies conditional on all nine clusters observed; no P values.",
        "S5_leave_one_patient_out": "Exploratory deletion of each distinct patient with complexity-model refitting. Rank frequencies and ranges are descriptive, not independent replications.",
        "S6_module_best_overlap": "Best Jaccard match of each primary module to a non-grey sensitivity module after C5/C8 exclusion at fixed genes and parameters. Matches can be many-to-one; not formal preservation testing.",
        "S7_module_all_overlaps": "All primary/non-grey sensitivity module pairs. Jaccard=intersection/union; fraction_primary_retained=intersection/primary size.",
        "S8_patient_module_tests": "Each sample-cluster requires >=10 cells; target minus equal mean of other eligible non-C5/C8 clusters in same sample, then equal average of sample contrasts per patient. Two-sided Wilcoxon only when n>=5. BH correction jointly across all 16 available P values; untested values remain missing.",
        "S9_patient_module_contrasts": "Underlying patient-level target-minus-reference mean standardized eigengene contrasts. n_paired_samples differs from n_patients.",
    }
    for name, frame in tables.items():
        frame.to_csv(out / (name + ".csv"), index=False, encoding="utf-8-sig")
    readme = pd.DataFrame([dict(sheet=name, description=descriptions[name]) for name in tables])
    with pd.ExcelWriter(out / "Supplementary_Tables_S1_S9_robustness.xlsx", engine="openpyxl") as writer:
        readme.to_excel(writer, sheet_name="README", index=False)
        for name, frame in tables.items():
            frame.to_excel(writer, sheet_name=name, index=False)
        for sheet in writer.book:
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                width = min(55, max(12, max(len(str(c.value or "")) for c in list(column)[:100]) + 2))
                sheet.column_dimensions[column[0].column_letter].width = width

    c1_loo = loo[loo.epithelial_cluster.eq("C1")]
    c1_boot = boot_summary[boot_summary.epithelial_cluster.eq("C1")].set_index("method")
    machine_summary = {
        "analysis_status": "exploratory_sensitivity_and_reproduction_of_existing_outputs",
        "seed": SEED, "bootstrap_replicates": N_BOOTSTRAP,
        "n_cells": len(cells), "n_patients": len(patients), "n_samples": int(cells.sample_id.nunique()),
        "patient_balancing_identifier": "patient_id",
        "patient_balancing_samples_rule": "Pool cells across samples within a patient and cluster before equal weighting patients",
        "sr_spearman_detected_genes": spearmanr(sr, cells.n_genes_by_counts).statistic,
        "sr_spearman_total_counts": spearmanr(sr, cells.total_counts).statistic,
        "complexity_model": {"formula": "SR ~ intercept + z(log1p(n_genes_by_counts)) + z(log1p(total_counts))", "coefficients": beta.tolist(), "r_squared": r_squared},
        "source_reproduction_checks": source_checks,
        "literal_cell_replication_bootstrap_checks": {"replicates_checked": len(direct_bootstrap_errors), "max_abs_difference": max(direct_bootstrap_errors)},
        "sample_module_means_independently_reproduced_from_cells": {"max_abs_difference": module_me_max_error},
        "c1_ranks": summary.loc["C1", [m + "_rank" for m in METHODS]].to_dict(),
        "complexity_adjusted_top_cluster": str(summary.complexity_adjusted_mean_SR_residual.idxmax()),
        "bootstrap_all_nine_clusters_observed": int(complete.sum()),
        "bootstrap_missing_at_least_one_cluster": int((~complete).sum()),
        "c1_bootstrap_rank1_fraction_conditional_complete": c1_boot.rank1_fraction_complete_replicates.to_dict(),
        "c1_leave_one_patient_out": {m: {
            "rank1_deletions": int((c1_loo.loc[c1_loo.method.eq(m), "rank_if_all_clusters_observed"] == 1).sum()),
            "evaluable_deletions": int(c1_loo.loc[c1_loo.method.eq(m), "rank_if_all_clusters_observed"].notna().sum()),
            "rank_min": float(c1_loo.loc[c1_loo.method.eq(m), "rank_if_all_clusters_observed"].min()),
            "rank_max": float(c1_loo.loc[c1_loo.method.eq(m), "rank_if_all_clusters_observed"].max())} for m in METHODS},
        "module_overlap_reproduced": True,
        "m1_overlap": best.set_index("primary_module").loc["M1"].to_dict(),
        "paired_module_tests_reproduced": True,
        "paired_module_fdr_family_size": int(tested.sum()),
        "paired_module_min_fdr": float(paired.fdr.min()),
        "paired_module_fdr_below_0_05_count": int(paired.fdr_below_0_05.sum()),
        "paired_module_n_patients_by_cluster": paired.groupby("cluster").n_patients.first().to_dict(),
        "interpretation": "C1 is highest for raw and patient-balanced SR but not after complexity adjustment; uneven coverage and patient-level module underpower require descriptive claims. Overlap is not a formal module-preservation test. Rank frequencies are empirical resampling stability, not posterior probabilities or P values.",
        "versions": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
        "input_manifest": {name: {"path_relative_to_root": path.relative_to(root).as_posix(), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for name, path in sources.items()},
    }
    (out / "robustness_summary.json").write_text(json.dumps(clean_value(machine_summary), indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    draw_figure(out, cells, summary, best, paired, boot_summary, machine_summary)
    write_text(out, summary, machine_summary, c1_boot)
    print(json.dumps(clean_value({k: v for k, v in machine_summary.items() if k not in ["versions", "input_manifest"]}), indent=2))


def draw_figure(out, cells, summary, best, paired, boot_summary, audit):
    plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.titlesize": 10,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.linewidth": 0.65, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})
    fig = plt.figure(figsize=(12.6, 9.0), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.02, 1.13], height_ratios=[1, 1], wspace=0.11, hspace=0.15)
    axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]
    for label, ax in zip("ABCDEF", axes):
        ax.text(-0.14, 1.07, label, transform=ax.transAxes, fontsize=15, fontweight="bold", va="bottom")
    ax = axes[0]
    h = ax.hexbin(cells.n_genes_by_counts, cells.signaling_entropy_rate, gridsize=42, mincnt=1, bins="log", cmap="viridis", linewidths=0, rasterized=True)
    ax.set(xlabel="Detected genes per cell", ylabel="Signaling entropy rate (SR)", title="SR depends on sequencing complexity")
    ax.text(0.03, 0.96, f"Spearman rho = {audit['sr_spearman_detected_genes']:.3f}\nComplexity model R² = {audit['complexity_model']['r_squared']:.3f}\nn = {len(cells):,} cells (descriptive)", transform=ax.transAxes, va="top", fontsize=7.5,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=3))
    cb = fig.colorbar(h, ax=ax, shrink=0.72, pad=0.02)
    cb.set_label("Cells per hexagon (log scale)", fontsize=7)

    ax = axes[1]
    values = summary[[m + "_rank" for m in METHODS]].to_numpy()
    ax.imshow(values, cmap="Blues_r", vmin=1, vmax=9, aspect="auto")
    ax.set_xticks(range(3), METHOD_LABELS)
    ax.set_yticks(range(9), [c + (" *" if c in ["C5", "C8"] else "") for c in CLUSTERS])
    for i in range(9):
        for j in range(3):
            ax.text(j, i, str(values[i, j]), ha="center", va="center", color="white" if values[i, j] <= 4 else "#20252B", fontweight="bold" if i == 1 else "normal", fontsize=9)
    ax.set(title="Cluster ranking changes after adjustment", xlabel="Rank 1 = highest cluster mean")
    ax.tick_params(length=0)
    ax.text(0, -0.21, "* C5: mixed identity; C8: immune-like", transform=ax.transAxes, fontsize=7)

    ax = axes[2]
    yy = np.arange(9)
    col = ["#C65338" if c == "C1" else "#385B78" for c in CLUSTERS]
    means = summary.patient_balanced_mean_SR.to_numpy()
    low, high = summary.patient_bootstrap_ci95_low.to_numpy(), summary.patient_bootstrap_ci95_high.to_numpy()
    for i in range(9):
        ax.plot([low[i], high[i]], [i, i], color=col[i], linewidth=1.7)
        ax.scatter(means[i], i, color=col[i], s=26, zorder=3)
    ax.set_yticks(yy, [f"{c}  (n = {summary.loc[c, 'n_patients']})" for c in CLUSTERS])
    ax.invert_yaxis()
    ax.set(xlabel="Patient-balanced SR (95% bootstrap interval)", title="Patients contribute equally within clusters", xlim=(0.848, 0.94))
    ax.grid(axis="x", color="#E1E6EB", linewidth=0.6)
    ax.text(0, -0.21, "n = patients represented in each cluster;\nintervals use whole-patient resampling.", transform=ax.transAxes, fontsize=7)

    ax = axes[3]
    ordered = best.set_index("primary_module").loc[MODULES]
    ax.barh(np.arange(8) - 0.17, ordered.jaccard, height=0.30, color="#385B78", label="Jaccard overlap")
    ax.barh(np.arange(8) + 0.17, ordered.fraction_primary_retained, height=0.30, color="#CF835A", label="Primary genes retained")
    ax.set_yticks(range(8), MODULES)
    ax.invert_yaxis()
    ax.set(xlim=(0, 1.06), xlabel="Fraction", title="Module overlap after excluding C5 and C8")
    ax.legend(loc="lower left", bbox_to_anchor=(0, -0.22), ncol=1, frameon=False, fontsize=7)
    ax.grid(axis="x", color="#E1E6EB", linewidth=0.6)
    ax.set_axisbelow(True)

    ax = axes[4]
    effects = paired.pivot(index="module", columns="cluster", values="paired_effect").reindex(index=MODULES, columns=CLUSTERS)
    n_patients = paired.groupby("cluster").n_patients.first()
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#E3E6E8")
    im = ax.imshow(effects, cmap=cmap, norm=TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5), aspect="auto")
    ax.set_yticks(range(8), MODULES)
    ax.set_xticks(range(9), [f"{c}\n{n_patients[c]}" for c in CLUSTERS])
    ax.tick_params(length=0)
    for i in range(8):
        for j in range(9):
            if np.isnan(effects.iloc[i, j]):
                ax.text(j, i, "—", fontsize=7, color="#73808A", ha="center", va="center")
    ax.set(xlabel="Cluster / paired patient n", title="Patient-level module contrasts")
    cb = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.025)
    cb.set_label("Mean paired standardized eigengene difference", fontsize=7)
    ax.text(0, -0.21, "Tests only for n ≥ 5; BH across 16 tests.\nNo FDR < 0.05 (minimum FDR = 0.125).", transform=ax.transAxes, fontsize=7)

    ax = axes[5]
    prob = boot_summary.pivot(index="epithelial_cluster", columns="method", values="rank1_fraction_complete_replicates").reindex(index=CLUSTERS, columns=METHODS)
    ax.imshow(prob, cmap="YlOrBr", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3), METHOD_LABELS)
    ax.set_yticks(range(9), CLUSTERS)
    ax.tick_params(length=0)
    for i in range(9):
        for j in range(3):
            value = prob.iloc[i, j]
            ax.text(j, i, f"{100 * value:.1f}%", ha="center", va="center", fontsize=7.5, color="white" if value > 0.65 else "#303030")
    ax.set(title="Exploratory patient bootstrap stability", xlabel="Fraction of resamples with rank 1")
    ax.text(0, -0.21, f"{audit['bootstrap_all_nine_clusters_observed']:,} / 10,000 resamples contained all 9 clusters.\nComplexity model refitted in every resample.", transform=ax.transAxes, fontsize=7)
    for ax in [axes[0], axes[2], axes[3]]:
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "Supplementary_Figure_S1_robustness.pdf", dpi=600)
    fig.savefig(out / "Supplementary_Figure_S1_robustness.png", dpi=600)
    fig.savefig(out / "Supplementary_Figure_S1_preview.png", dpi=150)
    plt.close(fig)


def write_text(out, summary, audit, c1_boot):
    loo = audit["c1_leave_one_patient_out"]
    text = f"""Supplementary methods

Sensitivity analyses were performed using the existing 5,609-cell epithelial-subset SR table, comprising {audit['n_patients']} distinct patients and {audit['n_samples']} samples. Patient-balanced cluster means gave each observed patient_id one equal contribution after pooling that patient's cells within the cluster across available samples. No sample_id was treated as an independent patient. Sequencing-complexity sensitivity used ordinary least squares regression of SR on an intercept, standardized log1p detected-gene counts, and standardized log1p total counts. Cluster means of the resulting residuals were ranked separately from unadjusted SR; residuals are not normalized entropy scores and were not used to redefine the trajectory root. These comparisons assess dependence on technical complexity, which can also reflect genuine biology, and do not establish that adjustment removes only technical variation.

As explicitly exploratory stability analyses added during manuscript revision, 10,000 bootstrap replicates resampled all {audit['n_patients']} patients with replacement (NumPy default_rng seed {SEED}), retaining all observed cells and samples of each sampled patient. In each replicate, we recalculated cell-weighted and patient-balanced means and refitted the complexity regression. Percentile intervals used replicates in which the corresponding cluster was observed. To keep rank comparisons on the same nine-cluster set, rank frequencies used only replicates containing all nine clusters ({audit['bootstrap_all_nine_clusters_observed']:,} of 10,000); the number of evaluable replicates is reported per endpoint. These frequencies quantify empirical resampling stability, not posterior probabilities or hypothesis-test P values. We additionally deleted each patient in turn and recalculated all three rankings, including regression refitting. These analyses were specified to audit existing claims, were not used to select a more favorable primary model, and cannot provide independent validation. Sparse patient coverage and possible non-random absence of clusters remain limitations.

For the existing C5/C8 exclusion network sensitivity, the analysis had reconstructed the network using the same 2,345 genes and soft-thresholding power of 7, with 490 metacells after exclusion. We independently recomputed all gene-set overlaps from the primary and sensitivity module assignments. Each primary module was matched to the non-grey sensitivity module with maximal Jaccard overlap (intersection divided by union); retained fraction was the intersection divided by primary module size. Best matches could be many-to-one. These descriptive overlaps do not constitute a formal WGCNA module-preservation test, and retention alone does not show preserved topology.

Patient-level module contrasts reproduced the existing analysis from sample-by-cluster mean standardized eigengenes. Each sample-cluster required at least 10 cells. Within a sample, the target cluster was compared with the unweighted mean of the other eligible clusters, excluding C5 and C8 from the reference. Sample contrasts were then averaged within each patient, giving one contrast per patient. C5 and C8 remained possible target clusters for transparency. A two-sided Wilcoxon signed-rank test was performed only when at least five paired patients and nonzero contrasts were available. Benjamini-Hochberg adjustment jointly covered all 16 available module-cluster P values. Untested P values and FDR values were kept missing, rather than labeled nonsignificant. Patient counts for these paired contrasts differ from the total number of patients with any cells in each cluster.

Supplementary results

SR was strongly associated with detected-gene counts (Spearman rho = {audit['sr_spearman_detected_genes']:.3f}) and total counts (rho = {audit['sr_spearman_total_counts']:.3f}); the two-predictor complexity model explained {100 * audit['complexity_model']['r_squared']:.1f}% of cell-level SR variation. C1 had the highest cell-weighted SR ({summary.loc['C1', 'cell_weighted_mean_SR']:.4f}) and patient-balanced SR ({summary.loc['C1', 'patient_balanced_mean_SR']:.4f}; 18 patients), but ranked fourth after complexity adjustment, for which C3 ranked first (Supplementary Figure S1A-C; Supplementary Table S1). Coverage was uneven: C0, C2 and C7 were represented by only 4, 3 and 3 patients, with {100 * summary.loc['C0', 'largest_patient_cell_fraction']:.1f}%, {100 * summary.loc['C2', 'largest_patient_cell_fraction']:.1f}% and {100 * summary.loc['C7', 'largest_patient_cell_fraction']:.1f}% of their cells, respectively, contributed by a single patient. Thus the raw C1 maximum does not establish a complexity-independent potency hierarchy.

In exploratory whole-patient bootstrap replicates with all nine clusters observed, C1 ranked first in {100 * c1_boot.loc['cell_weighted', 'rank1_fraction_complete_replicates']:.1f}% of cell-weighted analyses, {100 * c1_boot.loc['patient_balanced', 'rank1_fraction_complete_replicates']:.1f}% of patient-balanced analyses and {100 * c1_boot.loc['complexity_adjusted', 'rank1_fraction_complete_replicates']:.1f}% of complexity-adjusted analyses (Supplementary Figure S1F; Supplementary Table S4). On leave-one-patient-out analysis, C1 ranked first in {loo['cell_weighted']['rank1_deletions']}/{loo['cell_weighted']['evaluable_deletions']} cell-weighted, {loo['patient_balanced']['rank1_deletions']}/{loo['patient_balanced']['evaluable_deletions']} patient-balanced and {loo['complexity_adjusted']['rank1_deletions']}/{loo['complexity_adjusted']['evaluable_deletions']} complexity-adjusted analyses (Supplementary Table S5). These findings characterize sensitivity within the available cohort and are not external validation.

After C5/C8 exclusion, M1 retained 248 of 295 genes (84.1%) in its best-matching sensitivity module, with Jaccard overlap 0.581. Across modules, Jaccard overlaps ranged from 0.258 to 0.778 and retained fractions from 30.7% to 97.3% (Supplementary Figure S1D; Supplementary Tables S6-S7), indicating heterogeneous overlap rather than uniform network stability. The patient-paired C1 comparison included only four patients and was therefore descriptive. Of the 16 estimable tests across all clusters, none passed FDR < 0.05; the minimum FDR was 0.125 (Supplementary Figure S1E; Supplementary Tables S8-S9). Module-cluster relationships should accordingly be described as cell-level or descriptive associations, without claiming patient-level statistical confirmation.

Supplementary Figure S1 legend

Supplementary Figure S1. Sequencing-complexity, patient-sampling and module-identity sensitivity analyses. (A) Cell-level SR versus detected genes, displayed as a density hexbin plot. The correlation is descriptive; cells are not independent patients. R² refers to the model containing both log1p detected genes and log1p total counts. (B) Cluster ranks from cell-weighted SR, equal-patient-weighted SR and cell-weighted complexity-regression residuals; rank 1 denotes the highest mean. Asterisks flag C5 (mixed identity) and C8 (immune-like). (C) Patient-balanced cluster means and 95% percentile intervals from 10,000 whole-patient bootstrap replicates; labels give distinct patients with any cells in each cluster. (D) Jaccard overlap and fraction of primary genes retained in the best-matching non-grey module after C5/C8 exclusion; the mapping need not be one-to-one and is not a formal preservation test. (E) Mean paired differences in standardized module eigengenes after within-sample target-reference comparison and within-patient averaging. Labels give paired patient counts, which differ from panel C. Grey cells and dashes indicate unavailable contrasts; colors for fewer than five patients show descriptive effects without tests. Only 16 available tests entered the joint Benjamini-Hochberg correction, and none had FDR < 0.05. (F) Exploratory rank-1 frequencies among the {audit['bootstrap_all_nine_clusters_observed']:,} whole-patient bootstrap replicates containing all nine clusters; the complexity model was refitted per replicate. Frequencies are empirical stability measures and should not be interpreted as probabilities of biological truth.
"""
    (out / "Supplementary_Methods_Results_Figure_Legend.txt").write_text(text, encoding="utf-8")
    (out / "README.txt").write_text("This directory contains independently reproduced and exploratory manuscript-revision sensitivity analyses.\n\nRun: python run_robustness_revision.py --root /path/to/repository\n\nMain outputs: Supplementary_Figure_S1_robustness.pdf (vector panels with a rasterized density layer); Supplementary_Figure_S1_robustness.png (600 dpi); Supplementary_Tables_S1_S9_robustness.xlsx; matching CSV source tables; Supplementary_Methods_Results_Figure_Legend.txt; robustness_summary.json.\n\nThe source files remain unchanged. SHA-256 source hashes and software versions are stored in robustness_summary.json. Bootstrap source estimates and multiplicity weights are retained as compressed CSV files. The bootstrap is patient_id-based, and model coefficients are refitted in each replicate and each patient deletion. The original per-cluster independent bootstrap intervals were replaced here by clearly labeled whole-patient bootstrap intervals to support correlated cross-cluster stability comparisons. No inferential P values were calculated for bootstrap rankings. No new wet-lab or independent-cohort validation is implied.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
