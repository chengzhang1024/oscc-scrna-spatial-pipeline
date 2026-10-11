# Archival dataset files

Publish the following files together in a Zenodo dataset record:

1. integrated_3ds_annotated_paper9.h5ad: exact annotated single-cell atlas used downstream.
2. GSE208253_cleaned_deconvolved.h5ad: exact cleaned and deconvolved spatial object.
3. analysis_inputs_and_source_tables.zip: fixed network inputs/outputs, gene sets, niche assignments, source tables, display coordinates and original vector panels.
4. data_manifest.json: project-relative destination, byte count and SHA-256 of each archived file.
5. README.md: dataset description and source citations.

The h5ad objects are derived data, not the original GEO raw reads. They contain public study cell/spot identifiers and pseudonymous patient/sample labels. No clinical identifiers should be added to them. Verify source reuse/redistribution terms when publishing this dataset. Cite GEO accessions and original investigators; do not claim ownership of their data.

The archive is separate from the GitHub release to avoid large binary files in Git. Add the published data DOI to this README and the manuscript's data statement before submission. `restore_inputs.py` checks all listed hashes before analyses run.
