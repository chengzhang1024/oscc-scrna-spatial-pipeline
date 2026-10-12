# OSCC single-cell and spatial transcriptomic analysis

Code and source-data companion for Zhang Cheng's manuscript on a basal-adhesion epithelial expression program in oral squamous cell carcinoma.

## Data and analysis scope

| GEO accession | Data | Role |
|---|---|---|
| GSE181919 | single-cell RNA sequencing | discovery cohort; 53,633 retained cells |
| GSE164241 | single-cell RNA sequencing | healthy mucosal reference; 84,720 retained cells |
| GSE310797 | single-cell RNA sequencing | external cohort; 94,562 retained cells, including 24,000 epithelial cells |
| GSE208253 | Visium spatial transcriptomics | 23,029 spots in 12 sections from 10 patients |

GEO: https://www.ncbi.nlm.nih.gov/geo/ . The exact derived objects, fixed annotations, network results and plot inputs used in the manuscript are distributed as attachments to the versioned GitHub release: https://github.com/chengzhang1024/oscc-scrna-spatial-pipeline/releases/tag/v1.1.0 . See docs/DATA_ARCHIVE.md for downloading and checksum verification.

This release supports two distinct workflows: numerical verification from frozen source tables, and rerunning downstream analysis from the archived annotated objects. It does not present historical preprocessing as a newly verified raw-read-to-figure pipeline. Annotated objects preserve the actual analysis inputs. GSE310797 was previously inspected; replication is external exploratory support.

## Layout

* Root scripts 07–22: annotation, composition, Figure 1.
* Root scripts 29–38: epithelial states, CNV, entropy, Figure 3.
* scripts/spatial: spatial preparation, deconvolution, MISTy and Figure 2.
* scripts/40–44: official R hdWGCNA computation and Figure 4.
* scripts/56–59: fixed-signature replication, spatial support and final Figure 5.
* source_tables: unfiltered published source tables, fixed gene sets and plotting matrices.
* scripts/verify_publication.py: independent statistical check from distributed source tables.
* scripts/restore_inputs.py: safe restoration and checksum verification of downloaded archived objects.
* envs: focused environment requirements; docs: execution instructions and provenance.

Historical survival and discarded candidate analyses are not part of this manuscript's execution sequence. Script 51 contains the historical figure5 function but the current runner executes its Figure 1–4 functions only; final Figure 5 uses script 59.

## Quick verification (no large data download)

Install the core environment described in envs/README.md, then:
```
python scripts/verify_publication.py
```
This checks the disjoint fixed signatures; 24,000 cells / 16 patients; rank-residual correlations; patient-level medians, exact two-sided sign-test p-values and BH q-values; min100 sensitivity; and the saved 100-set random baseline. It does not equate recomputing stored summaries with rerunning the raw analysis.

## Rerun from archived inputs

Download the data attachments listed in docs/DATA_ARCHIVE.md into a separate directory. First extract integrated_3ds_annotated_paper9.h5ad.zip there; leave analysis_inputs_and_source_tables.zip compressed. Then:
```
python scripts/restore_inputs.py /path/to/downloaded/files
python scripts/56_figure5_program_replication.py all
python scripts/59_plot_figure5_visual_redesign.py
```
Run from this repository root. Figures, data and results are local outputs excluded from Git. The archived inputs contain the published UMAP and display matrices; script59 reuses these frozen matrices. Script58 can recompute display coordinates but is not required to reproduce published coordinates.

Further commands and conditions are in docs/EXECUTION.md. Original hashes and packaging changes are in docs/script_provenance.json and the data manifest. Do not replace the frozen data with a newly annotated cohort and expect identical results.

## License and citation

Original analysis code is MIT licensed. Public source datasets retain their original terms. Third-party software and gene sets retain their own licenses; source data are redistributed only where their source terms permit. Cite the original cohort publications and the software used. Author/release metadata: CITATION.cff.

## Release preparation

Repository portability fixes, documentation, packaging and verification helpers were prepared with assistance from OpenAI Codex. The author is responsible for reviewing the release. Original numerical source data and scientific manuscript paragraphs were not changed during preparation.
