# Execution and verification

Always run from the repository root. Restore archived data first. Do not run every numbered script automatically: annotation scripts can overwrite upstream objects and historical plot functions may not belong to this manuscript.

## Verified downstream workflow
1. `python scripts/verify_publication.py` checks the frozen distributed tables.
2. `python scripts/restore_inputs.py DOWNLOAD_DIRECTORY` restores the archived project tree with SHA-256 verification.
3. `python scripts/56_figure5_program_replication.py all` reselects discovery genes, computes external patient associations and mixed-spot support, and validates numerical inputs.
4. `python scripts/59_plot_figure5_visual_redesign.py` draws final Figure 5 from frozen coordinates and display tables. Do not run script58 unless deliberately regenerating UMAP coordinates.
5. `python scripts/reproduce_figures.py` assembles the current Figure 1–4 revision and final Figure5; numerical source tables are not changed intentionally.

## Other analysis stages
Figure 3 scripts29–38 use the archived atlas and prior fixed epithelial cell tables. Figures1–4 are redrawn from frozen results with script51. Figure4: script40 creates R inputs, `Rscript scripts/41_run_figure4_hdwgcna.R` computes the official network, script42 plots, and script44 validates. The original network outputs are in the data archive; full R retraining is distinct from reproducing those original plots. Use a dedicated output copy if refitting.

Figure2: prepare_gse208253.py and train_deconvolution.py document preprocessing/training; the published final spatial object is archived. MISTy is run by run_figure2c_misty.py and run_figure2i_misty.py; final panels are plot_figure2a through plot_figure2l. GPU model retraining and MISTy fitting can be time-consuming and version-sensitive. This release does not claim these training stages have been freshly rerun during packaging.

## Reproducibility boundaries
Patient and section identifiers are pseudonymous labels from public studies. Discovery gene selection uses five eligible paired patients. External cells are 24,000 from 16 patients; one patient has32 cells and is removed by min100 sensitivity. Mixed spatial spots do not identify epithelial-intrinsic gene expression. The data archive freezes annotations and niches so downstream numerical differences can be diagnosed separately from retraining differences.
