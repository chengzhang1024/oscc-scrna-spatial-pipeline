"""Run Figure 2A-G with the GSE181919/Visium paper9 taxonomy."""

from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_pipeline():
    path = ROOT / "08_fig2_literature_v3.py"
    spec = importlib.util.spec_from_file_location("fig2_base", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def main():
    pipeline = load_pipeline()
    pipeline.H5AD = ROOT / "results/integrated_3ds_annotated_paper9.h5ad"
    pipeline.RESULT = ROOT / "results/paper9"
    pipeline.FIGURE = ROOT / "figures/paper9"
    pipeline.PDF_FIGURE = pipeline.FIGURE / "pdf"
    pipeline.CELL_TYPES = [
        "B/Plasma cells", "Dendritic cells", "Endothelial cells",
        "Epithelial cells", "Fibroblasts", "Macrophages", "Mast cells",
        "Myocytes", "NK/T cells",
    ]
    pipeline.COMMON_TYPES = pipeline.CELL_TYPES
    pipeline.CELL_COLORS = {
        "B/Plasma cells": "#4E9F8A",
        "Dendritic cells": "#8A6BB8",
        "Endothelial cells": "#4AA6A8",
        "Epithelial cells": "#D95F59",
        "Fibroblasts": "#6A9F58",
        "Macrophages": "#C28E2C",
        "Mast cells": "#C05A87",
        "Myocytes": "#B06B3C",
        "NK/T cells": "#3B78B5",
    }
    pipeline.main()


if __name__ == "__main__":
    main()
