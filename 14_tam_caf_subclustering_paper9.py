"""Run TAM/CAF screening with paper9 Macrophages and Fibroblasts."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_pipeline():
    path = ROOT / "10_tam_caf_subclustering.py"
    spec = importlib.util.spec_from_file_location("tam_caf_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    pipeline = load_pipeline()
    pipeline.H5AD = ROOT / "results/integrated_3ds_annotated_paper9.h5ad"
    pipeline.CELL_TYPE_FIELD = "cell_type_paper9"
    pipeline.RESULT = ROOT / "results/paper9"
    pipeline.FIGURE = ROOT / "figures/paper9"
    pipeline.PDF_FIGURE = pipeline.FIGURE / "pdf"
    pipeline.CONFIGS["Myeloid"]["parent"] = "Macrophages"
    pipeline.main()


if __name__ == "__main__":
    main()
