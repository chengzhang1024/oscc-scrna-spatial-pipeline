Sys.setlocale("LC_ALL", "English_United States.utf8")
Sys.setenv(LC_ALL = "English_United States.utf8", LANG = "English_United States.utf8",
           OMP_NUM_THREADS = "2", OPENBLAS_NUM_THREADS = "2")
root <- normalizePath(getwd(), winslash = "/")
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
# Load hdWGCNA without attaching enrichR, whose startup hook performs a web request.
# All enrichment is computed locally from the versioned GMT file.
suppressPackageStartupMessages({
  library(Seurat); library(WGCNA); library(Matrix); library(dplyr)
  attachNamespace(loadNamespace("hdWGCNA"))
})
options(future.globals.maxSize = 2 * 1024^3)
future::plan("sequential")
WGCNA::disableWGCNAThreads()
foreach::registerDoSEQ()
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) RhpcBLASctl::blas_set_num_threads(2)
set.seed(20260905)
out_rel <- "results/paper9/figure4_epithelial_hdwgcna"
out <- file.path(root, out_rel)
input <- file.path(out, "input")
dir.create(out, recursive = TRUE, showWarnings = FALSE)
logfile <- file(file.path(out, "run.log"), "at")
sink(logfile, split = TRUE)
message("Figure 4: official hdWGCNA ", packageVersion("hdWGCNA"))
write_csv <- function(x, name, row_names = FALSE) {
  write.csv(x, file.path(out, name), row.names = row_names, fileEncoding = "UTF-8")
}
cells <- read.csv(file.path(input, "cells.csv"), stringsAsFactors = FALSE)
rownames(cells) <- cells$r_cell_id
clusters <- paste0("C", 0:8)
prepared <- file.path(out, "prepared.rds")
network_file <- file.path(out, "network.rds")
complete_file <- file.path(out, "analysis.rds")

if (!file.exists(prepared)) {
  counts <- as(readMM(file.path(input, "counts.mtx")), "CsparseMatrix")
  genes <- read.csv(file.path(input, "genes.csv"))$gene
  dimnames(counts) <- list(genes, cells$r_cell_id)
  stopifnot(ncol(counts) == 5609L, !anyDuplicated(genes))
  obj <- CreateSeuratObject(counts, meta.data = cells, min.cells = 0, min.features = 0)
  obj <- NormalizeData(obj, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
  hvg <- read.csv(file.path(root, "results/paper9/figure3_epithelial/Fig3A_epithelial_hvg_genes.csv"))$gene
  detected <- Matrix::rowMeans(counts > 0)
  network_genes <- intersect(hvg, names(detected)[detected >= 0.05])
  write_csv(data.frame(gene = genes, fraction_cells = detected,
                       patient_balanced_HVG = genes %in% hvg,
                       selected = genes %in% network_genes), "gene_selection.csv")
  VariableFeatures(obj) <- intersect(hvg, rownames(obj))
  obj <- ScaleData(obj, features = VariableFeatures(obj), verbose = FALSE)
  obj <- RunPCA(obj, features = VariableFeatures(obj), npcs = 30, verbose = FALSE, seed.use = 20260905)
  umap <- as.matrix(cells[, c("umap_1", "umap_2")])
  rownames(umap) <- cells$r_cell_id; colnames(umap) <- c("UMAP_1", "UMAP_2")
  obj[["umap"]] <- CreateDimReducObject(umap, key = "UMAP_", assay = "RNA")
  obj <- SetupForWGCNA(obj, gene_select = "custom", gene_list = network_genes, wgcna_name = "epithelial")
  message("Metacells: same sample and cluster; PCA; k=15, max_shared=7, min_cells=23")
  obj <- MetacellsByGroups(obj, group.by = c("sample_id", "epithelial_cluster"),
                          ident.group = "epithelial_cluster", reduction = "pca", dims = 1:15,
                          k = 15, max_shared = 7, min_cells = 23, target_metacells = 50,
                          max_iter = 5000, mode = "sum", verbose = FALSE)
  obj <- NormalizeMetacells(obj, normalization.method = "LogNormalize", scale.factor = 10000, verbose = FALSE)
  mc <- GetMetacellObject(obj)
  mm <- mc@meta.data
  mm$metacell_id <- rownames(mm)
  members <- do.call(rbind, lapply(seq_len(nrow(mm)), function(i) {
    data.frame(metacell_id = mm$metacell_id[i], r_cell_id = strsplit(mm$cells_merged[i], ",", fixed = TRUE)[[1]])
  }))
  members <- merge(members, cells[, c("r_cell_id", "cell_id", "sample_id", "patient_id", "disease_status", "epithelial_cluster")], by = "r_cell_id", sort = FALSE)
  membership_audit <- aggregate(members[, c("sample_id", "patient_id", "disease_status", "epithelial_cluster")],
                                list(metacell_id = members$metacell_id), function(x) length(unique(x)))
  stopifnot(all(as.matrix(membership_audit[, -1]) == 1))
  write_csv(members, "metacell_membership.csv")
  write_csv(membership_audit, "metacell_identity_audit.csv")
  write_csv(mm, "metacell_metadata.csv")
  write_csv(data.frame(r_cell_id = cells$r_cell_id, entered_metacell = cells$r_cell_id %in% members$r_cell_id), "cell_network_coverage.csv")
  obj <- SetDatExpr(obj, group.by = "epithelial_cluster", group_name = clusters, assay = "RNA", layer = "data")
  rm(counts, mc); gc()
  saveRDS(obj, prepared, compress = FALSE)
} else {
  obj <- readRDS(prepared)
}

if (!file.exists(network_file)) {
  message("Soft-power diagnostics using WGCNA::pickSoftThreshold")
  obj <- TestSoftPowers(obj, corFnc = "cor", networkType = "signed", blockSize = 1000)
  powers <- GetPowerTable(obj)
  # The reference plot uses 0.8, although its prose says 0.9; export both thresholds.
  eligible <- powers$Power[powers$SFT.R.sq >= 0.8 & powers$slope < 0 & powers$mean.k. >= 1]
  if (!length(eligible)) stop("No usable scale-free threshold; inspect power diagnostics before proceeding")
  beta <- min(eligible)
  powers$selected <- powers$Power == beta
  write_csv(powers, "soft_power.csv")
  message("Constructing signed network, beta=", beta, "; genes=", ncol(GetDatExpr(obj)))
  obj <- ConstructNetwork(obj, soft_power = beta, corType = "pearson", randomSeed = 20260905,
                          tom_outdir = paste0(out_rel, "/TOM"), tom_name = "epithelial",
                          overwrite_tom = TRUE, maxBlockSize = 5000,
                          deepSplit = 4, minModuleSize = 50, mergeCutHeight = 0.2,
                          pamStage = FALSE, detectCutHeight = 0.995, verbose = 2)
  write_csv(GetModules(obj), "modules_before_connectivity.csv")
  saveRDS(obj, network_file, compress = FALSE)
} else {
  obj <- readRDS(network_file)
}

if (!file.exists(complete_file)) {
  message("Computing single-cell module eigengenes with the official implementation")
  obj <- ModuleEigengenes(obj, group.by.vars = NULL, verbose = FALSE, npcs = 5)
  obj <- ModuleConnectivity(obj, harmonized = FALSE, reassign_modules = FALSE, sparse = TRUE)
  # Keep the package's grey category and signed kME; change names only for display.
  obj <- ResetModuleNames(obj, new_name = "M")
  obj <- RunModuleUMAP(obj, n_hubs = 10, n_neighbors = 25, min_dist = 0.15,
                       supervised = FALSE, n_threads = 2, n_sgd_threads = 1, verbose = TRUE)
  saveRDS(obj, complete_file, compress = FALSE)
} else {
  obj <- readRDS(complete_file)
}
modules <- GetModules(obj)
write_csv(modules, "gene_modules.csv")
write_csv(GetHubGenes(obj, n_hubs = 100), "hub_genes.csv")
mes <- GetMEs(obj, harmonized = FALSE)
write_csv(data.frame(r_cell_id = rownames(mes), mes, check.names = FALSE), "cell_module_eigengenes.csv")
write_csv(GetModuleUMAP(obj), "module_umap.csv")
dat <- GetDatExpr(obj)
write_csv(data.frame(metacell_id = rownames(dat), dat, check.names = FALSE), "metacell_expression.csv")
tom <- GetTOM(obj)
con <- file(file.path(out, "tom_float32.bin"), "wb")
writeBin(as.numeric(tom), con, size = 4, endian = "little"); close(con)
write_csv(data.frame(gene = rownames(tom)), "tom_gene_order.csv")
net <- GetNetworkData(obj)
pdf(file.path(out, "module_dendrogram_QC.pdf"), width = 10, height = 4, useDingbats = FALSE)
WGCNA::plotDendroAndColors(net$dendrograms[[1]], net$colors, "Modules", dendroLabels = FALSE, hang = 0.03)
dev.off()
metadata <- list(implementation = "R hdWGCNA", hdWGCNA_version = as.character(packageVersion("hdWGCNA")),
                 Seurat_version = as.character(packageVersion("Seurat")), WGCNA_version = as.character(packageVersion("WGCNA")),
                 cohort = "GSE181919", n_cells = nrow(cells), n_samples = length(unique(cells$sample_id)),
                 n_patients = length(unique(cells$patient_id)), n_metacells = nrow(dat), n_genes = ncol(dat),
                 metacell_grouping = c("sample_id", "epithelial_cluster"), k = 15, max_shared = 7,
                 min_cells = 23, target_metacells_per_group = 50, reduction = "PCA 1:15",
                 gene_selection = "Figure 3 patient-balanced 3000 HVGs intersect >=5% single-cell detection, with full-library normalization",
                 soft_power = GetWGCNAParams(obj)$power, module_counts = as.list(table(modules$module)),
                 grey_policy = "unassigned genes retained as grey, no forced reassignment",
                 eigengenes = "Seurat-scaled single-cell RNA data, unharmonized first PC; patient-level contrasts exported separately",
                 network_layout = "official unsupervised RunModuleUMAP on TOM; no imposed intermodule bridges",
                 seed = 20260905)
jsonlite::write_json(metadata, file.path(out, "analysis_metadata.json"), pretty = TRUE, auto_unbox = TRUE)
writeLines(capture.output(sessionInfo()), file.path(out, "sessionInfo.txt"))
message("Official hdWGCNA analysis and exports complete")
sink(); close(logfile)
