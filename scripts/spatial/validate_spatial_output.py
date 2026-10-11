from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse


FINAL_PATH = ROOT / "data" / "processed" / "GSE208253" / "GSE208253_cleaned_deconvolved.h5ad"
RESULT_DIR = ROOT / "results" / "spatial_preprocessing"
MODEL_ROOT = ROOT / "models" / "destvi"

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

MARKERS = {
    "B/Plasma cells": ["CD79A", "MS4A1", "MZB1"],
    "Dendritic cells": ["FCER1A", "CD1C", "CLEC10A"],
    "Endothelial cells": ["PECAM1", "VWF", "KDR"],
    "Epithelial cells": ["EPCAM", "KRT19", "KRT14"],
    "Fibroblasts": ["COL1A1", "DCN", "COL3A1"],
    "Macrophages": ["LST1", "TYROBP", "C1QC"],
    "Mast cells": ["KIT", "TPSAB1", "CPA3"],
    "Myocytes": ["DES", "MYH11", "ACTA2"],
    "NK/T cells": ["CD3D", "CD3E", "TRBC1"],
}


def matrix_mean(matrix) -> np.ndarray:
    return np.asarray(matrix.mean(axis=1)).ravel() if sparse.issparse(matrix) else matrix.mean(axis=1)


def main() -> None:
    adata = sc.read_h5ad(FINAL_PATH)
    counts = adata.layers["counts"]
    count_values = counts.data if sparse.issparse(counts) else counts.ravel()
    normalized_values = adata.X.data if sparse.issparse(adata.X) else adata.X.ravel()
    c2l = pd.DataFrame(
        adata.obsm["cell2location_proportions"], index=adata.obs_names, columns=CELL_TYPES
    )
    destvi = pd.DataFrame(
        adata.obsm["destvi_proportions"], index=adata.obs_names, columns=CELL_TYPES
    )

    checks = {
        "shape_is_23029_by_36601": adata.shape == (23029, 36601),
        "twelve_samples": adata.obs["sample_id"].nunique() == 12,
        "ten_patients": adata.obs["patient_id"].nunique() == 10,
        "raw_counts_are_nonnegative_integers": bool(
            (count_values >= 0).all() and np.allclose(count_values, np.round(count_values))
        ),
        "normalized_X_is_nonnegative_finite": bool(
            (normalized_values >= 0).all() and np.isfinite(normalized_values).all()
        ),
        "cell2location_is_finite": bool(np.isfinite(c2l.to_numpy()).all()),
        "cell2location_rows_sum_to_one": bool(np.allclose(c2l.sum(axis=1), 1.0, atol=1e-6)),
        "destvi_is_finite": bool(np.isfinite(destvi.to_numpy()).all()),
        "destvi_rows_sum_to_one": bool(np.allclose(destvi.sum(axis=1), 1.0, atol=1e-5)),
        "twelve_spatial_images": len(adata.uns["spatial"]) == 12,
        "twelve_destvi_models": all(
            (MODEL_ROOT / f"spatial_model_S{number}" / "model.pt").exists()
            for number in range(1, 13)
        ),
    }

    marker_rows = []
    for cell_type, markers in MARKERS.items():
        available = [gene for gene in markers if gene in adata.var_names]
        score = matrix_mean(adata[:, available].X)
        marker_rows.append(
            {
                "cell_type": cell_type,
                "markers": ";".join(available),
                "cell2location_spearman_r": float(
                    pd.Series(score).corr(pd.Series(c2l[cell_type].to_numpy()), method="spearman")
                ),
                "destvi_proportion_spearman_r": float(
                    pd.Series(score).corr(pd.Series(destvi[cell_type].to_numpy()), method="spearman")
                ),
            }
        )
    marker_path = RESULT_DIR / "deconvolution_marker_validation.csv"
    pd.DataFrame(marker_rows).to_csv(marker_path, index=False)

    report = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "n_spots": int(adata.n_obs),
        "n_genes": int(adata.n_vars),
        "cell_types": CELL_TYPES,
        "marker_validation": str(marker_path),
        "interpretation": "Cell-type-specific expression from DestVI is model-inferred.",
    }
    output_path = RESULT_DIR / "validation_report.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise RuntimeError("Final spatial output validation failed")


if __name__ == "__main__":
    main()
