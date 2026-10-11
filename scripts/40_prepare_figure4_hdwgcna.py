"""Export the fixed Figure 3 cohort with all available raw-count genes for R."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import anndata as ad
from scipy import sparse, io

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/paper9/figure4_epithelial_hdwgcna/input"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cells = pd.read_csv(ROOT / "results/paper9/figure3_epithelial/Fig3A_epithelial_cells_and_umap.csv.gz")
    assert len(cells) == 5609 and cells.cell_id.is_unique
    cells["qc_identity"] = cells.epithelial_cluster.map({"C5": "mixed_identity", "C8": "immune_like"}).fillna("epithelial")
    source = ad.read_h5ad(ROOT / "results/integrated_3ds_annotated_paper9.h5ad", backed="r")
    rows = source.obs_names.get_indexer(cells.cell_id)
    assert np.all(rows >= 0)
    counts = source.layers["counts"][rows, :]
    if hasattr(counts, "to_memory"):
        counts = counts.to_memory()
    counts = sparse.csr_matrix(counts)
    genes = source.var_names.astype(str).to_numpy()
    source.file.close()
    assert np.all(counts.data >= 0) and np.allclose(counts.data, np.rint(counts.data))
    # Stable ASCII IDs avoid R's automatic barcode/name sanitization.
    cells["r_cell_id"] = [f"EPI{i:05d}" for i in range(len(cells))]
    cells.to_csv(OUT / "cells.csv", index=False)
    pd.DataFrame({"gene": genes}).to_csv(OUT / "genes.csv", index=False)
    with (OUT / "counts.mtx").open("wb") as handle:
        io.mmwrite(handle, counts.T.astype(np.int32), symmetry="general")
    groups = cells.groupby(["sample_id", "patient_id", "disease_status", "epithelial_cluster"], observed=True).size().rename("n_cells").reset_index()
    groups.to_csv(OUT / "sample_cluster_sizes.csv", index=False)
    audit = {"n_cells": len(cells), "n_genes": len(genes), "n_patients": cells.patient_id.nunique(),
             "n_samples": cells.sample_id.nunique(), "stages": cells.disease_status.value_counts().to_dict(),
             "input_policy": "Fixed Figure 3 cohort and labels, including C5/C8 as previously requested; raw counts for all available genes; no 1200-gene cap."}
    (OUT / "input_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2), flush=True)
    fraction = counts.getnnz(axis=0) / counts.shape[0]
    print({str(t): int(np.sum(fraction >= t)) for t in (0.05, 0.10, 0.15, 0.20)}, flush=True)
    print(groups[groups.disease_status.eq("OLK")].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
