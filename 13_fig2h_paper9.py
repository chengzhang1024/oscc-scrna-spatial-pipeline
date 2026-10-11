"""Run Figure 2H pseudobulk analysis with the paper9 taxonomy."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def load_pipeline():
    path = ROOT / "09_fig2h_literature_v3.py"
    spec = importlib.util.spec_from_file_location("fig2h_base", path)
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

    cache_root = Path(os.environ["USERPROFILE"]) / ".cache/gseapy"

    def load_cached_gene_sets(library, metabolic_only=False):
        path = cache_root / f"Enrichr.{library}.gmt"
        if not path.exists():
            raise FileNotFoundError(f"Missing local gene-set cache: {path}")
        output = {}
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                fields = line.rstrip("\n").split("\t")
                if len(fields) >= 3:
                    output[fields[0]] = {gene for gene in fields[2:] if gene}
        if metabolic_only:
            output = {
                term: genes for term, genes in output.items()
                if any(keyword in term.lower() for keyword in pipeline.METABOLIC_KEYWORDS)
            }
        print(f"  loaded {len(output)} cached pathways from {library}", flush=True)
        return output

    pipeline.load_gene_sets = load_cached_gene_sets
    # This environment intermittently aborts inside Matplotlib after the KEGG
    # matrix is written. Plotting is performed by 15_plot_fig2h_paper9.py with
    # the stable system Matplotlib environment.
    pipeline.plot_heatmap = lambda *args, **kwargs: None
    pipeline.main()


if __name__ == "__main__":
    main()
