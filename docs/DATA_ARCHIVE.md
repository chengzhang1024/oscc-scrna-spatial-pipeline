# Derived inputs distributed with the GitHub release

Release: https://github.com/chengzhang1024/oscc-scrna-spatial-pipeline/releases/tag/v1.1.0

Download these attachments into one directory:

1. integrated_3ds_annotated_paper9.h5ad.zip: losslessly compressed annotated single-cell atlas. Extract it in the download directory to obtain integrated_3ds_annotated_paper9.h5ad.
2. GSE208253_cleaned_deconvolved.h5ad: exact cleaned and deconvolved spatial object.
3. analysis_inputs_and_source_tables.zip: fixed network inputs/outputs, gene sets, niche assignments, source tables, display coordinates and original vector panels. Leave this ZIP compressed.
4. data_manifest.json: original project-relative paths, byte counts and SHA-256 checksums of all 277 archived inputs.
5. DATA_README.md: dataset description, source citations and restoration instructions.
6. release_asset_checksums.json: size and SHA-256 of the distributed attachments, including the ZIP wrapper.

Extract the code ZIP separately or clone the repository. From its root, run:

```text
python scripts/restore_inputs.py /path/to/downloaded/files
```

The command verifies every original input against data_manifest.json. The single-cell ZIP wrapper preserves the original H5AD byte-for-byte; its extraction was checked against the original SHA-256.

These H5AD objects are derived inputs, not original GEO raw reads. They contain public study cell/spot identifiers and pseudonymous sample labels. Source datasets, software and gene sets retain their original rights, attribution and licensing terms. Cite the original GEO studies. The MIT code license does not change original source-data rights.

Large binary objects are release attachments, not Git-tracked files. A Zenodo DOI is not assigned to this release.
