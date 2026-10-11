from __future__ import annotations

import gzip
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("NUMBA_CACHE_DIR", str(ROOT / ".cache" / "numba"))

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from PIL import Image
from scipy import sparse


RAW_DIR = ROOT / "data" / "raw" / "GSE208253"
OUT_DIR = ROOT / "data" / "intermediate" / "GSE208253"
RESULT_DIR = ROOT / "results" / "spatial_preprocessing"
MANIFEST_PATH = RAW_DIR / "sample_manifest.csv"
OUTPUT_PATH = OUT_DIR / "GSE208253_cleaned_pre_deconvolution.h5ad"

POSITION_COLUMNS = [
    "barcode",
    "in_tissue",
    "array_row",
    "array_col",
    "pxl_row_in_fullres",
    "pxl_col_in_fullres",
]


def robust_lower(values: np.ndarray, absolute_floor: int) -> int:
    logged = np.log1p(values.astype(float))
    median = np.median(logged)
    mad = np.median(np.abs(logged - median))
    adaptive = int(np.floor(np.expm1(median - 3.5 * max(mad, 0.05))))
    return max(absolute_floor, adaptive)


def robust_mt_upper(values: np.ndarray) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return min(30.0, max(20.0, median + 4.0 * max(mad, 0.5)))


def read_gzip_json(path: Path) -> dict:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def read_gzip_image(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as handle:
        image = Image.open(handle).convert("RGB")
        return np.asarray(image, dtype=np.uint8)


def normalize_log1p(counts: sparse.spmatrix) -> sparse.csr_matrix:
    totals = np.asarray(counts.sum(axis=1)).ravel()
    factors = np.divide(1e4, totals, out=np.zeros_like(totals, dtype=float), where=totals > 0)
    normalized = sparse.diags(factors).dot(counts).tocsr().astype(np.float32)
    normalized.data = np.log1p(normalized.data)
    return normalized


def locate_files(row: pd.Series) -> dict[str, Path]:
    prefix = f"{row.gsm}_{row.library_id}"
    files = {
        "counts": RAW_DIR / f"{prefix}_filtered_feature_bc_matrix.h5",
        "positions": RAW_DIR / f"{prefix}_tissue_positions_list.csv.gz",
        "scalefactors": RAW_DIR / f"{prefix}_scalefactors_json.json.gz",
        "image": RAW_DIR / f"{prefix}_tissue_hires_image.png.gz",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for {row.sample_id}: {missing}")
    return files


def prepare_sample(row: pd.Series) -> tuple[ad.AnnData, dict, pd.DataFrame]:
    files = locate_files(row)
    adata = sc.read_10x_h5(files["counts"], gex_only=True)
    adata.var_names_make_unique()

    positions = pd.read_csv(files["positions"], header=None, names=POSITION_COLUMNS)
    positions = positions.set_index("barcode")
    if not adata.obs_names.isin(positions.index).all():
        missing = adata.obs_names[~adata.obs_names.isin(positions.index)].tolist()[:5]
        raise ValueError(f"Coordinate barcodes missing for {row.sample_id}: {missing}")
    positions = positions.reindex(adata.obs_names)
    if not (positions["in_tissue"].to_numpy() == 1).all():
        raise ValueError(f"Filtered matrix contains off-tissue spots for {row.sample_id}")

    counts = adata.X.tocsr()
    total_counts = np.asarray(counts.sum(axis=1)).ravel()
    n_genes = np.asarray((counts > 0).sum(axis=1)).ravel()
    mt_mask = np.asarray(adata.var_names.str.upper().str.startswith("MT-"), dtype=bool)
    mt_counts = np.asarray(counts[:, mt_mask].sum(axis=1)).ravel()
    pct_mt = np.divide(
        mt_counts * 100.0,
        total_counts,
        out=np.zeros_like(mt_counts, dtype=float),
        where=total_counts > 0,
    )

    min_counts = robust_lower(total_counts, absolute_floor=500)
    min_genes = robust_lower(n_genes, absolute_floor=500)
    max_pct_mt = robust_mt_upper(pct_mt)
    keep = (total_counts >= min_counts) & (n_genes >= min_genes) & (pct_mt <= max_pct_mt)

    qc_spots = pd.DataFrame(
        {
            "sample_id": row.sample_id,
            "patient_id": row.patient_id,
            "barcode": adata.obs_names,
            "total_counts": total_counts,
            "n_genes_by_counts": n_genes,
            "pct_counts_mt": pct_mt,
            "pass_qc": keep,
        }
    )

    retained = int(keep.sum())
    retained_fraction = float(keep.mean())
    sample_pass = bool(
        retained >= 500
        and retained_fraction >= 0.70
        and np.median(n_genes[keep]) >= 1000
        and np.median(total_counts[keep]) >= 1000
    )

    summary = {
        "sample_id": row.sample_id,
        "gsm": row.gsm,
        "patient_id": row.patient_id,
        "stage": row.stage,
        "spots_before": int(adata.n_obs),
        "spots_after": retained,
        "retained_fraction": retained_fraction,
        "min_counts": min_counts,
        "min_genes": min_genes,
        "max_pct_mt": max_pct_mt,
        "median_counts_before": float(np.median(total_counts)),
        "median_genes_before": float(np.median(n_genes)),
        "median_pct_mt_before": float(np.median(pct_mt)),
        "median_counts_after": float(np.median(total_counts[keep])) if retained else np.nan,
        "median_genes_after": float(np.median(n_genes[keep])) if retained else np.nan,
        "median_pct_mt_after": float(np.median(pct_mt[keep])) if retained else np.nan,
        "sample_pass": sample_pass,
        "exclusion_reason": "" if sample_pass else "Failed retained spot/fraction or median complexity criteria",
    }

    adata = adata[keep].copy()
    counts = adata.X.tocsr().astype(np.int32)
    adata.layers["counts"] = counts.copy()
    adata.X = normalize_log1p(counts)

    sample_positions = positions.iloc[np.flatnonzero(keep)]
    adata.obs["sample_id"] = row.sample_id
    adata.obs["gsm"] = row.gsm
    adata.obs["patient_id"] = row.patient_id
    adata.obs["stage"] = row.stage
    adata.obs["tissue"] = row.tissue
    adata.obs["dataset"] = row.dataset
    adata.obs["array_row"] = sample_positions["array_row"].to_numpy(dtype=int)
    adata.obs["array_col"] = sample_positions["array_col"].to_numpy(dtype=int)
    adata.obs["pxl_row_in_fullres"] = sample_positions["pxl_row_in_fullres"].to_numpy(dtype=float)
    adata.obs["pxl_col_in_fullres"] = sample_positions["pxl_col_in_fullres"].to_numpy(dtype=float)
    adata.obs["total_counts"] = total_counts[keep]
    adata.obs["n_genes_by_counts"] = n_genes[keep]
    adata.obs["pct_counts_mt"] = pct_mt[keep]
    adata.obs["pass_qc"] = True
    adata.obs_names = [f"{row.sample_id}_{barcode}" for barcode in adata.obs_names]
    adata.obs_names.name = "spot_id"

    adata.var["gene_symbol"] = adata.var_names.astype(str)
    adata.var["mt"] = mt_mask
    adata.obsm["spatial"] = np.column_stack(
        [
            sample_positions["pxl_col_in_fullres"].to_numpy(dtype=float),
            sample_positions["pxl_row_in_fullres"].to_numpy(dtype=float),
        ]
    )
    scalefactors = read_gzip_json(files["scalefactors"])
    image = read_gzip_image(files["image"])
    adata.uns["spatial"] = {
        row.sample_id: {
            "images": {"hires": image},
            "scalefactors": scalefactors,
            "metadata": {
                "source": "GSE208253",
                "gsm": row.gsm,
                "patient_id": row.patient_id,
                "stage": row.stage,
                "image_source": files["image"].name,
            },
        }
    }
    return adata, summary, qc_spots


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST_PATH, dtype=str)

    samples: list[ad.AnnData] = []
    summaries: list[dict] = []
    spot_tables: list[pd.DataFrame] = []
    for row in manifest.itertuples(index=False):
        print(f"Preparing {row.sample_id} ({row.gsm})", flush=True)
        sample, summary, spots = prepare_sample(row)
        summaries.append(summary)
        spot_tables.append(spots)
        if summary["sample_pass"]:
            samples.append(sample)

    report = pd.DataFrame(summaries)
    report.to_csv(RESULT_DIR / "qc_report.csv", index=False)
    pd.concat(spot_tables, ignore_index=True).to_csv(
        RESULT_DIR / "qc_metrics_by_spot.csv.gz", index=False, compression="gzip"
    )
    manifest = manifest.merge(
        report[["sample_id", "sample_pass", "exclusion_reason"]], on="sample_id", how="left"
    )
    manifest.to_csv(RESULT_DIR / "sample_manifest.csv", index=False)

    if not samples:
        raise RuntimeError("No samples passed QC")
    combined = ad.concat(
        samples,
        axis=0,
        join="inner",
        merge="same",
        uns_merge="unique",
        index_unique=None,
    )
    combined.uns["processing"] = {
        "dataset": "GSE208253",
        "normalization": "per-sample total-count normalization to 1e4 followed by log1p",
        "raw_counts_layer": "counts",
        "qc": "sample-specific robust lower count/gene thresholds and robust mitochondrial upper threshold",
        "batch_correction": "none",
        "analysis_scope": "within-patient scoring",
    }
    combined.write_h5ad(OUTPUT_PATH, compression="gzip")
    print(f"Saved {OUTPUT_PATH}: {combined.n_obs} spots x {combined.n_vars} genes", flush=True)


if __name__ == "__main__":
    main()
