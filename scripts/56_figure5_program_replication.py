"""Discovery-frozen epithelial program replication and mixed-spot spatial support.

Run with .envs/figure5_py/Scripts/python.exe -X utf8; independent of survival scripts.
"""
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pathlib import Path
import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
import warnings

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, binomtest
from patsy import dmatrix

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper9/figure5_program_replication"
SC = ROOT / "results/integrated_3ds_annotated_paper9.h5ad"
SP = ROOT / "data/processed/GSE208253/GSE208253_cleaned_deconvolved.h5ad"
DISC = ROOT / "results/paper9/figure4_epithelial_hdwgcna/input/cells.csv"
MOD = ROOT / "results/paper9/figure4_epithelial_hdwgcna/gene_modules.csv"
STAGE = ROOT / "results/paper9/literature_v3_GSE310797_confirmed_stage_mapping.csv"
SEED = 20260916
BASAL = ["KRT5", "KRT14", "KRT15", "COL17A1", "ITGA6"]
CYCLE = ["MKI67", "TOP2A", "PCNA", "MCM2", "MCM4", "TYMS"]
QC = ["total_counts_available", "n_detected_available", "pct_mt_available"]


def dump(name, obj):
    (OUT / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def bh(p):
    p = np.asarray(p, float)
    q = np.full(len(p), np.nan)
    ok = np.flatnonzero(np.isfinite(p))
    order = ok[np.argsort(p[ok])]
    q[order] = np.minimum(1, np.minimum.accumulate((p[order] * len(ok) / np.arange(1, len(ok)+1))[::-1])[::-1])
    return q


def summarize(values, key, seed=SEED):
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return dict(comparison=key, n_patients=0, median=np.nan, ci_low=np.nan, ci_high=np.nan, n_positive=0, p=np.nan)
    rng = np.random.default_rng(seed)
    med = np.median(rng.choice(v, (10000, len(v)), replace=True), axis=1)
    nz = v[v != 0]
    return dict(comparison=key, n_patients=len(v), median=float(np.median(v)),
                ci_low=float(np.quantile(med, .025)) if len(v)>1 else np.nan,
                ci_high=float(np.quantile(med, .975)) if len(v)>1 else np.nan,
                n_positive=int((v > 0).sum()),
                p=float(binomtest(int((nz > 0).sum()), len(nz), .5).pvalue) if len(nz) else 1.)


def extract_sc(cell_ids, keep_genes=None):
    """Read original CSR counts in sorted row blocks; normalize before column selection."""
    a = ad.read_h5ad(SC, backed="r")
    genes = a.var_names.astype(str).to_numpy()
    rows = a.obs_names.get_indexer(cell_ids)
    assert (rows >= 0).all() and len(set(rows)) == len(rows)
    a.file.close()
    col = np.arange(len(genes)) if keep_genes is None else pd.Index(genes).get_indexer(keep_genes)
    assert (col >= 0).all()
    sort = np.argsort(rows)
    blocks, qcs = [], []
    mt = np.char.startswith(genes.astype(str), "MT-")
    with h5py.File(SC, "r") as f:
        g = f["layers/counts"]
        ptr = g["indptr"][:]
        for start in range(0, len(rows), 512):
            rr = rows[sort[start:start+512]]
            lo, hi = int(rr.min()), int(rr.max())+1
            b, e = int(ptr[lo]), int(ptr[hi])
            raw = sparse.csr_matrix((g["data"][b:e], g["indices"][b:e], ptr[lo:hi+1]-b),
                                    shape=(hi-lo, len(genes)))[rr-lo].astype(np.float32)
            raw.eliminate_zeros()
            totals = np.asarray(raw.sum(axis=1)).ravel().astype(float)
            assert np.all(totals > 0) and np.all(raw.data >= 0)
            qcs.append(np.c_[totals, np.diff(raw.indptr), np.asarray(raw[:, mt].sum(axis=1)).ravel()*100/totals])
            x = raw.multiply((1e4/totals)[:, None]).tocsr()
            x.data = np.log1p(x.data).astype(np.float32)
            blocks.append(x[:, col])
    inverse = np.argsort(sort)
    return sparse.vstack(blocks, format="csr")[inverse], pd.DataFrame(np.vstack(qcs)[inverse], columns=QC), genes[col]


def marker_table(x, obs, genes, omit=None):
    deltas, detects, differences, used = [], [], [], []
    for patient, frame in obs.groupby("patient_id", sort=True):
        if patient == omit:
            continue
        a = frame.index[frame.epithelial_cluster.eq("C1")].to_numpy()
        b = frame.index[~frame.epithelial_cluster.isin(["C1", "C5", "C8"])].to_numpy()
        if len(a) < 10 or len(b) < 10:
            continue
        ma = np.asarray(x[a].mean(axis=0)).ravel()
        mb = np.asarray(x[b].mean(axis=0)).ravel()
        da = np.asarray((x[a] > 0).mean(axis=0)).ravel()
        db = np.asarray((x[b] > 0).mean(axis=0)).ravel()
        deltas.append(ma-mb); detects.append(da); differences.append(da-db); used.append(patient)
    assert len(used) >= 4
    delta = np.stack(deltas)
    t = pd.DataFrame(dict(gene=genes, paired_mean_delta=delta.mean(axis=0),
                          positive_patient_fraction=(delta > 0).mean(axis=0),
                          mean_C1_detection=np.mean(detects, axis=0), mean_detection_delta=np.mean(differences, axis=0)))
    t["marker_score"] = t.paired_mean_delta * t.mean_detection_delta.clip(lower=0)
    technical = t.gene.str.match(r"^(MT-|RPL|RPS|IGH|IGK|IGL|TRAV|TRBV)")
    t["eligible"] = (~technical & t.mean_C1_detection.ge(.2) & t.positive_patient_fraction.ge(.7)
                     & t.paired_mean_delta.gt(0) & t.mean_detection_delta.gt(0))
    return t, used, delta


def discovery():
    obs = pd.read_csv(DISC).reset_index(drop=True)
    x, qc, genes = extract_sc(obs.cell_id)
    obs = pd.concat([obs, qc], axis=1)
    obs.to_csv(OUT / "discovery_cells.csv.gz", index=False)
    sparse.save_npz(OUT / "discovery_expression.npz", x)
    pd.Series(genes, name="gene").to_csv(OUT / "sc_gene_universe.csv", index=False)
    table, patients, delta = marker_table(x, obs, genes)
    table.to_csv(OUT / "discovery_all_gene_marker_audit.csv", index=False)
    selected = table[table.eligible].sort_values(["marker_score", "gene"], ascending=[False, True]).head(60).copy()
    if len(selected) < 20:
        raise RuntimeError("Fewer than 20 patient-consistent candidate genes; review required, no automatic relaxation")
    selected["rank"] = np.arange(1, len(selected)+1)
    selected["split"] = np.where(selected["rank"] % 2 == 1, "C1_score", "C1_check")
    selected["control_gene_excluded"] = selected.gene.isin(BASAL+CYCLE)
    selected.to_csv(OUT / "discovery_selected_markers.csv", index=False)
    paircounts = pd.crosstab(obs.patient_id, obs.epithelial_cluster)
    paircounts["other_eligible"] = paircounts.drop(columns=["C1", "C5", "C8"]).sum(axis=1)
    paircounts["used_for_marker_selection"] = paircounts.index.isin(patients)
    paircounts.to_csv(OUT / "discovery_patient_coverage.csv")
    pd.DataFrame(delta.T, index=genes, columns=patients).loc[selected.gene].to_csv(OUT / "selected_gene_patient_deltas.csv")
    sensitivity = []
    for patient in patients:
        t, _, _ = marker_table(x, obs, genes, omit=patient)
        ranking = t[t.eligible].sort_values(["marker_score", "gene"], ascending=[False, True]).copy()
        ranking["rank_without_patient"] = np.arange(1, len(ranking)+1)
        v = selected[["gene"]].merge(t, on="gene").merge(ranking[["gene", "rank_without_patient"]], on="gene", how="left")
        v["omitted_patient"] = patient
        sensitivity.append(v)
    pd.concat(sensitivity).to_csv(OUT / "discovery_leave_one_patient_out.csv", index=False)
    definitions = {s: selected.loc[selected.split.eq(s) & ~selected.control_gene_excluded, "gene"].tolist()
                   for s in ["C1_score", "C1_check"]}
    definitions["C1_full"] = selected.gene.tolist()
    definitions["Basal_control"] = BASAL
    definitions["Cycle_control"] = CYCLE
    mods = pd.read_csv(MOD)
    for m in ["M1", "M5"]:
        definitions[m+"_full"] = mods.loc[mods.module.eq(m), "gene_name"].tolist()
        definitions[m] = [g for g in definitions[m+"_full"] if g not in set(selected.gene)]
    # Null checking signatures matched using discovery-only mean expression and detection.
    background = pd.DataFrame(dict(gene=genes, mean=np.asarray(x.mean(axis=0)).ravel(),
                                   detection=np.asarray((x > 0).mean(axis=0)).ravel()))
    background["mbin"] = pd.qcut(background["mean"].rank(method="first"), 8, labels=False)
    background["dbin"] = pd.qcut(background.detection.rank(method="first"), 8, labels=False)
    background = background.set_index("gene")
    pool = background.loc[~background.index.isin(selected.gene.tolist()+BASAL+CYCLE)].copy()
    pool = pool.loc[~pool.index.to_series().str.match(r"^(MT-|RPL|RPS|IGH|IGK|IGL|TRAV|TRBV)")]
    rng = np.random.default_rng(SEED)
    nullsets = {}
    for i in range(100):
        chosen = []
        for gene in definitions["C1_check"]:
            dist = (pool.mbin-background.loc[gene, "mbin"]).abs() + (pool.dbin-background.loc[gene, "dbin"]).abs()
            dist.loc[pool.index.isin(chosen)] = 1000
            options = dist.index[dist.eq(dist.min())].to_numpy()
            chosen.append(str(rng.choice(options)))
        nullsets[f"null_{i:03d}"] = chosen
    dump("signatures.json", definitions)
    dump("matched_null_signatures.json", nullsets)
    pd.DataFrame([dict(signature=s, gene=g, order=j+1) for s, gs in definitions.items() for j, g in enumerate(gs)]).to_csv(OUT / "frozen_signature_genes.csv", index=False)
    dump("frozen_manifest.json", dict(seed=SEED, discovery_paired_patients=patients,
         signature_sizes={k: len(v) for k, v in definitions.items()},
         file_sha256={str(p.relative_to(ROOT)): sha(p) for p in [DISC, MOD, OUT/"analysis_plan.md", OUT/"signatures.json", OUT/"matched_null_signatures.json"]},
         scoring="Mean log1p CP10k, available shared genes; no cohort z-score before primary scoring",
         previous_data_exposure="GSE310797 had previously been analyzed; exploratory extension, not pristine confirmatory validation"))
    print("DISCOVERY FROZEN", patients, {k: len(v) for k, v in definitions.items()}, flush=True)


def scores_from(x, genes, definitions):
    lookup = pd.Index(genes)
    scores, audit = {}, []
    for s, gg in definitions.items():
        ix = lookup.get_indexer(gg)
        present = ix[ix >= 0]
        if not len(present):
            raise ValueError(f"No measurable genes for {s}")
        scores[s] = np.asarray(x[:, present].mean(axis=1)).ravel()
        scores[s+"_detected"] = np.asarray((x[:, present] > 0).mean(axis=1)).ravel()
        audit.append(dict(signature=s, n_defined=len(gg), n_available=len(present), missing=";".join(np.asarray(gg)[ix < 0])))
    return pd.DataFrame(scores), pd.DataFrame(audit)


def design_rank(frame, biological=False):
    names = QC + (["Basal_control", "Cycle_control"] if biological else [])
    vals = rankdata(frame[names].to_numpy(float), axis=0)
    vals -= vals.mean(axis=0)
    sd = vals.std(axis=0)
    vals = vals[:, sd > 1e-10] / sd[sd > 1e-10]
    return np.c_[np.ones(len(frame)), vals]


def residual_ranks(values, design):
    r = rankdata(values, axis=0)
    return r - design @ np.linalg.lstsq(design, r, rcond=None)[0]


def correlate_residual(x, y):
    x = np.asarray(x).reshape(-1)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, None]
    x = x-x.mean()
    y = y-y.mean(axis=0)
    denom = np.linalg.norm(x)*np.linalg.norm(y, axis=0)
    return np.divide(x@y, denom, out=np.full(y.shape[1], np.nan), where=denom > 1e-12)


def single_cell():
    signatures = json.loads((OUT / "signatures.json").read_text(encoding="utf-8"))
    nullsets = json.loads((OUT / "matched_null_signatures.json").read_text(encoding="utf-8"))
    source = ad.read_h5ad(SC, backed="r")
    obs = source.obs.copy()
    source.file.close()
    obs = obs[obs.dataset.astype(str).eq("B5_GSE310797") & obs.cell_type_paper9.astype(str).eq("Epithelial cells")].copy()
    obs.insert(0, "cell_id", obs.index.astype(str))
    obs = obs[["cell_id", "patient_id", "sample_id", "disease_status"]].reset_index(drop=True)
    obs = obs.merge(pd.read_csv(STAGE)[["patient_id", "patient", "stage_group"]], on="patient_id", validate="many_to_one")
    assert len(obs) == 24000 and obs.patient_id.nunique() == 16
    genes = sorted(set(g for gg in list(signatures.values())+list(nullsets.values()) for g in gg))
    x, qc, genes = extract_sc(obs.cell_id, genes)
    scores, coverage = scores_from(x, genes, signatures)
    nullscores, _ = scores_from(x, genes, nullsets)
    frame = pd.concat([obs, qc, scores], axis=1)
    frame.to_csv(OUT / "replication_cell_scores.csv.gz", index=False)
    coverage.to_csv(OUT / "replication_gene_coverage.csv", index=False)
    rows, gene_rows, null_rows = [], [], []
    check_genes = signatures["C1_check"]
    check_expr = x[:, pd.Index(genes).get_indexer(check_genes)].toarray()
    nullmat = nullscores[list(nullsets)].to_numpy()
    for patient, f in frame.groupby("patient", sort=True):
        ids = f.index.to_numpy()
        q = design_rank(f)
        b = design_rank(f, biological=True)
        c1 = residual_ranks(f.C1_score.to_numpy(), q)
        c1b = residual_ranks(f.C1_score.to_numpy(), b)
        tests = {
            "C1_check_QC": correlate_residual(c1, residual_ranks(f.C1_check.to_numpy(), q))[0],
            "C1_check_QC_biology": correlate_residual(c1b, residual_ranks(f.C1_check.to_numpy(), b))[0],
            "C1_M1": correlate_residual(c1, residual_ranks(f.M1.to_numpy(), q))[0],
            "C1_M5": correlate_residual(c1, residual_ranks(f.M5.to_numpy(), q))[0]}
        for name, rho in tests.items():
            rows.append(dict(patient=patient, patient_id=f.patient_id.iloc[0], n_cells=len(f), comparison=name, rho=rho))
        gene_rho = correlate_residual(c1, residual_ranks(check_expr[ids], q))
        for gene, rho in zip(check_genes, gene_rho):
            gene_rows.append(dict(patient=patient, gene=gene, rho=rho, n_cells=len(f)))
        null_rho = correlate_residual(c1b, residual_ranks(nullmat[ids], b))
        for name, rho in zip(nullsets, null_rho):
            null_rows.append(dict(patient=patient, null_signature=name, rho=rho))
    effects = pd.DataFrame(rows)
    effects.to_csv(OUT / "replication_patient_effects.csv", index=False)
    pd.DataFrame(gene_rows).to_csv(OUT / "replication_check_gene_correlations.csv", index=False)
    nulltab = pd.DataFrame(null_rows)
    nulltab.to_csv(OUT / "replication_matched_null_correlations.csv", index=False)
    summ = pd.DataFrame([summarize(g.rho, name) for name, g in effects.groupby("comparison")])
    summ["bh_q"] = bh(summ.p)
    summ.to_csv(OUT / "replication_summary.csv", index=False)
    sens = pd.DataFrame([summarize(g.rho, name) for name, g in effects[effects.n_cells.ge(100)].groupby("comparison")])
    sens["bh_q"] = bh(sens.p)
    sens.to_csv(OUT / "replication_min100_sensitivity.csv", index=False)
    observed = float(summ.set_index("comparison").loc["C1_check_QC_biology", "median"])
    null_medians = nulltab.groupby("null_signature").rho.median()
    dump("specificity_baseline.json", dict(observed_patient_median=observed,
         null_median=float(null_medians.median()), null_95th=float(null_medians.quantile(.95)),
         empirical_upper_tail=(1+int((null_medians >= observed).sum()))/(1+len(null_medians)),
         interpretation="Discovery-matched random checking-set baseline; not a test of malignant specificity"))
    columns = ["C1_score", "C1_check", "M1", "M5"]
    ps = frame.groupby(["patient", "patient_id"])[columns].agg(["median", "mean"])
    ps.columns = ["_".join(c) for c in ps.columns]
    ps["n_cells"] = frame.groupby(["patient", "patient_id"]).size()
    ps.to_csv(OUT / "replication_patient_scores.csv")
    disc = pd.read_csv(OUT / "discovery_cells.csv.gz")
    dx = sparse.load_npz(OUT / "discovery_expression.npz")
    dg = pd.read_csv(OUT / "sc_gene_universe.csv").gene.to_numpy()
    ds, _ = scores_from(dx, dg, signatures)
    disc = pd.concat([disc, ds], axis=1)
    disc.to_csv(OUT / "discovery_signature_scores.csv.gz", index=False)
    disc.groupby(["patient_id", "epithelial_cluster", "disease_status"])[columns].median().to_csv(OUT / "discovery_patient_cluster_scores.csv")
    print("REPLICATION", summ.to_string(index=False), flush=True)


def standardized(a):
    a = np.asarray(a, float)
    if a.ndim == 1:
        a = a[:, None]
    a = a-a.mean(axis=0)
    sd = a.std(axis=0)
    return a[:, sd > 1e-10] / sd[sd > 1e-10]


def fit_spatial(f, niche, signature, mode):
    abundance = "epithelial_proportion" if mode == "proportion" else "epithelial_abundance"
    a = f[f.niche_id.eq("N1")]
    b = f[f.niche_id.eq(niche)]
    base = dict(sample_id=f.sample_id.iloc[0], patient_id=f.patient_id.iloc[0], signature=signature,
                contrast=niche+"-N1", mode=mode, n_reference_all=len(a), n_target_all=len(b))
    if min(len(a), len(b)) < 30:
        return dict(**base, eligible=False, reason="<30 spots in one niche before support restriction")
    lower = max(a[abundance].quantile(.1), b[abundance].quantile(.1))
    upper = min(a[abundance].quantile(.9), b[abundance].quantile(.9))
    sel = f[f.niche_id.isin(["N1", niche]) & f[abundance].between(lower, upper)].copy()
    n0 = int(sel.niche_id.eq("N1").sum()); n1 = int(sel.niche_id.eq(niche).sum())
    base.update(support_low=lower, support_high=upper, n_reference=n0, n_target=n1,
                retained_fraction=len(sel)/(len(a)+len(b)))
    if upper <= lower or min(n0, n1) < 30:
        return dict(**base, eligible=False, reason="Insufficient epithelial-abundance common support")
    spline = np.asarray(dmatrix("bs(v, df=4, degree=3, include_intercept=False)-1",
                                  {"v": np.log1p(sel[abundance].to_numpy())}))
    nuis = np.c_[spline, np.log1p(sel["total_counts_available"]),
                 np.log1p(sel["n_detected_available"]), sel["pct_mt_available"]]
    if mode == "spatial_trend":
        xy = standardized(sel[["x", "y"]].to_numpy())
        nuis = np.c_[nuis, xy, xy[:, 0]**2, xy[:, 1]**2, xy[:, 0]*xy[:, 1]]
    design = np.c_[np.ones(len(sel)), sel.niche_id.eq(niche).astype(float), standardized(nuis)]
    # Standardize against the complete section, not contrast-specific selected spots.
    sd = f[signature].std(ddof=0)
    if sd <= 1e-10:
        return dict(**base, eligible=False, reason="Constant signature in section")
    y = (sel[signature].to_numpy()-f[signature].mean())/sd
    coef, _, rank, _ = np.linalg.lstsq(design, y, rcond=None)
    condition = float(np.linalg.cond(design))
    if rank != design.shape[1] or condition > 1e8:
        return dict(**base, eligible=False, reason="Rank-deficient/ill-conditioned design", condition=condition)
    return dict(**base, eligible=True, reason="", coefficient=float(coef[1]),
                rank=int(rank), condition=condition, n_parameters=design.shape[1])


def spatial_analysis():
    signatures = json.loads((OUT / "signatures.json").read_text(encoding="utf-8"))
    a = ad.read_h5ad(SP)
    raw = a.layers["counts"].tocsr().astype(np.float32)
    assert raw.shape[0] == 23029 and np.all(raw.data >= 0)
    genes = a.var_names.astype(str).to_numpy()
    universe = pd.read_csv(OUT / "sc_gene_universe.csv").gene
    common = pd.Index(genes).intersection(universe)
    ix = pd.Index(genes).get_indexer(common)
    raw = raw[:, ix]
    totals = np.asarray(raw.sum(axis=1)).ravel().astype(float)
    raw.eliminate_zeros()
    mt = np.array([g.startswith("MT-") for g in common])
    qc = pd.DataFrame({QC[0]: totals, QC[1]: np.diff(raw.indptr),
                        QC[2]: np.asarray(raw[:, mt].sum(axis=1)).ravel()*100/totals})
    log = raw.multiply((1e4/totals)[:, None]).tocsr()
    log.data = np.log1p(log.data).astype(np.float32)
    scores, coverage = scores_from(log, common, signatures)
    coverage.to_csv(OUT / "spatial_gene_coverage.csv", index=False)
    obs = a.obs[["sample_id", "patient_id", "stage"]].reset_index().rename(columns={a.obs.index.name or "index": "spot_id"})
    celltypes = list(a.uns["deconvolution"]["cell_types"])
    epi = celltypes.index("Epithelial cells")
    scores["epithelial_abundance"] = np.asarray(a.obsm["cell2location_abundance"])[:, epi]
    scores["epithelial_proportion"] = np.asarray(a.obsm["cell2location_proportions"])[:, epi]
    scores["x"] = np.asarray(a.obsm["spatial"])[:, 0]
    scores["y"] = np.asarray(a.obsm["spatial"])[:, 1]
    frame = pd.concat([obs, qc, scores], axis=1)
    early = pd.read_csv(ROOT / "figures/paper9/figure2/Fig2F_niche_assignments_early_OSCC.csv.gz")
    late = pd.read_csv(ROOT / "figures/paper9/figure2/Fig2L_niche_assignments_advanced_OSCC.csv.gz")
    late = late.rename(columns={"projected_niche_id": "niche_id", "projected_niche_name": "niche_name"})
    early["out_of_distribution"] = False
    assign = pd.concat([early, late], ignore_index=True)
    assert assign.spot_id.is_unique
    frame = frame.merge(assign[["spot_id", "niche_id", "niche_name", "out_of_distribution"]], on="spot_id", validate="one_to_one")
    assert len(frame) == 23029 and frame.patient_id.nunique() == 10
    assert frame.niche_id.isin(["N1", "N2", "N3"]).all() and frame.niche_name.notna().all()
    frame.to_csv(OUT / "spatial_spot_scores.csv.gz", index=False)
    spatial_groups = frame.groupby(["patient_id", "sample_id", "niche_id"], observed=True)
    spatial_groups[["C1_score", "M1", "M5", "epithelial_abundance"]].median().to_csv(OUT / "spatial_niche_descriptive_scores.csv")
    rows = []
    for sample, f in frame.groupby("sample_id", sort=True, observed=True):
        for mode in ["primary", "spatial_trend", "proportion"]:
            for niche in ["N2", "N3"]:
                for sig in ["C1_score", "M1", "M5"]:
                    rows.append(fit_spatial(f, niche, sig, mode))
        print("SPATIAL", sample, len(f), flush=True)
    section = pd.DataFrame(rows)
    section.to_csv(OUT / "spatial_section_effects.csv", index=False)
    valid = section[section.eligible]
    patients = valid.groupby(["mode", "patient_id", "signature", "contrast"]).agg(
        coefficient=("coefficient", "mean"), n_sections=("sample_id", "nunique")).reset_index()
    patients.to_csv(OUT / "spatial_patient_effects.csv", index=False)
    summary = []
    for (mode, sig, contrast), f in patients.groupby(["mode", "signature", "contrast"]):
        r = summarize(f.coefficient, sig+":"+contrast)
        r.update(mode=mode, signature=sig, contrast=contrast, sufficient_patients=len(f) >= 6)
        summary.append(r)
    summary = pd.DataFrame(summary)
    summary["bh_q"] = np.nan
    for mode in summary["mode"].unique():
        mask = summary["mode"].eq(mode)
        summary.loc[mask, "bh_q"] = bh(summary.loc[mask, "p"])
    summary.to_csv(OUT / "spatial_summary.csv", index=False)
    dump("spatial_processing.json", dict(n_spots=len(frame), n_patients=frame.patient_id.nunique(),
         n_sections=frame.sample_id.nunique(), normalization_background=len(common),
         representative_sections=["S6", "S2"], representative_rule="Existing Fig2 selections; not new signature values",
         no_spot_level_pvalues=True, no_epithelial_intrinsic_claim=True,
         input_sha256={str(p.relative_to(ROOT)): sha(p) for p in [SP]}))
    print(summary[summary['mode'].eq('primary')].to_string(index=False), flush=True)


def validate():
    sig = json.loads((OUT / "signatures.json").read_text(encoding="utf-8"))
    assert not (set(sig["C1_score"]) & set(sig["C1_check"]))
    assert not (set(sig["C1_full"]) & (set(sig["M1"]) | set(sig["M5"])))
    assert not ((set(sig["C1_score"]) | set(sig["C1_check"])) & set(BASAL+CYCLE))
    scores = pd.read_csv(OUT / "replication_cell_scores.csv.gz")
    spots = pd.read_csv(OUT / "spatial_spot_scores.csv.gz")
    effects = pd.read_csv(OUT / "replication_patient_effects.csv")
    assert len(scores) == 24000 and scores.cell_id.is_unique and scores.patient_id.nunique() == 16
    assert len(spots) == 23029 and spots.spot_id.is_unique and spots.patient_id.nunique() == 10
    assert spots.niche_id.isin(["N1", "N2", "N3"]).all()
    assert effects.groupby("comparison").size().eq(16).all()
    assert np.isfinite(scores[["C1_score", "C1_check", "M1", "M5"]]).all().all()
    assert np.isfinite(effects.rho).all()
    for path in ["replication_gene_coverage.csv", "spatial_gene_coverage.csv"]:
        cov = pd.read_csv(OUT / path)
        assert (cov.n_available / cov.n_defined >= .8).all()
    # Independently reproduce a residual association via the precision matrix.
    f = scores[scores.patient.eq(scores.patient.iloc[0])]
    arr = rankdata(f[["C1_score", "C1_check"]+QC].to_numpy(), axis=0)
    inv = np.linalg.pinv(np.corrcoef(arr, rowvar=False))
    independent = -inv[0, 1]/np.sqrt(inv[0, 0]*inv[1, 1])
    saved = effects[(effects.patient.eq(f.patient.iloc[0])) & effects.comparison.eq("C1_check_QC")].rho.iloc[0]
    assert np.isclose(saved, independent, atol=1e-8)
    sections = pd.read_csv(OUT / "spatial_section_effects.csv")
    assert len(sections) == 12*3*2*3
    good = sections[sections.eligible]
    assert (good.n_reference.ge(30) & good.n_target.ge(30)).all()
    patients = pd.read_csv(OUT / "spatial_patient_effects.csv")
    check = good.groupby(["mode", "patient_id", "signature", "contrast"]).coefficient.mean()
    assert np.allclose(patients.set_index(["mode", "patient_id", "signature", "contrast"]).coefficient.reindex(check.index), check)
    # Frozen inputs unchanged since discovery, before external results.
    manifest = json.loads((OUT / "frozen_manifest.json").read_text(encoding="utf-8"))
    assert all(sha(ROOT / Path(p.replace("\\", "/"))) == value for p, value in manifest["file_sha256"].items())
    report = dict(status="passed", checks=["disjoint gene sets", "complete patient/spot coverage", ">=80% gene availability",
          "partial correlation independent precision-matrix check", "patient aggregation", "common support", "frozen hashes"],
          spatial_failed_section_contrasts=int((~sections.eligible).sum()),
          discovery_eligible_patients=manifest["discovery_paired_patients"],
          limitations=["Only five discovery patients have >=10 cells in both comparison groups",
                       "Previously inspected external cohort; exploratory program support",
                       "Mixed-spot localization, not epithelial-specific or causal expression",
                       "No untouched cohort validation, malignancy validation, or network preservation claim"],
          software={p: importlib.metadata.version(p) for p in ["anndata", "numpy", "pandas", "scipy", "matplotlib", "patsy"]},
          python=platform.python_version(), script_sha256=sha(__file__))
    dump("validation_report.json", report)
    print("VALIDATION PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["discovery", "single_cell", "spatial", "validate", "all"])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, func in [("discovery", discovery), ("single_cell", single_cell), ("spatial", spatial_analysis), ("validate", validate)]:
        if args.stage in [name, "all"]:
            func()
