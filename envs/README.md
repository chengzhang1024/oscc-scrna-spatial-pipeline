# Environments

Use separate environments for replication/figures (Python 3.13), spatial training (Python 3.10), and hdWGCNA (R). Run commands from the repository root.

Core replication:
```
python -m venv .venv
# Windows: .venv\Scripts\activate; Linux/macOS: source .venv/bin/activate
python -m pip install -r envs/requirements-core.txt
python scripts/verify_publication.py
```
For frozen Figure 1–5 redrawing: `python -m pip install -r envs/requirements-redraw.txt`. The broader optional analysis dependencies are in requirements-figures.txt.
For spatial training: create `envs/spatial.yml`. Install the appropriate PyTorch build for your hardware separately when using a GPU. The original training used GPU acceleration; CPU training may be slow and floating-point differences are possible.

The R package version record is `figure4_R_packages.csv`. Install Seurat 5.5.1, WGCNA 1.74 and hdWGCNA 0.4.12 with their dependencies in a separate R library. These scripts do not automatically install R packages or modify a system library. R version and session details are provided with the archived source tables. Exact retraining of metacells and networks can depend on R/package versions; the original network outputs are archived.

Pins record the analysis environment, not a claim that every platform has binary wheels. Core and figure verification results are in docs/release_validation.json. Historical upstream training is documented separately from verified postprocessing.
