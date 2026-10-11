Sys.setlocale("LC_ALL", "English_United States.utf8")
root <- normalizePath(getwd(), winslash = "/")
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages({
  library(Seurat); library(WGCNA); library(dplyr)
  attachNamespace(loadNamespace("hdWGCNA"))
})
WGCNA::disableWGCNAThreads(); foreach::registerDoSEQ()
if (requireNamespace("RhpcBLASctl", quietly = TRUE)) RhpcBLASctl::blas_set_num_threads(2)
set.seed(20260905)
out_rel <- "results/paper9/figure4_epithelial_hdwgcna"
out <- file.path(root, out_rel)
obj <- readRDS(file.path(out, "prepared.rds"))
obj <- SetDatExpr(obj, group.by = "epithelial_cluster", group_name = paste0("C", c(0:4,6,7)),
                  assay = "RNA", layer = "data")
beta <- jsonlite::read_json(file.path(out, "analysis_metadata.json"))$soft_power
obj <- ConstructNetwork(obj, soft_power = beta, corType = "pearson", randomSeed = 20260905,
                        tom_outdir = paste0(out_rel, "/TOM_sensitivity_without_C5_C8"), tom_name = "sensitivity",
                        overwrite_tom = TRUE, maxBlockSize = 5000, deepSplit = 4, minModuleSize = 50,
                        mergeCutHeight = 0.2, pamStage = FALSE, detectCutHeight = 0.995, verbose = 1)
sens <- GetModules(obj)
write.csv(sens, file.path(out, "sensitivity_without_C5_C8_gene_modules.csv"), row.names = FALSE)
primary <- read.csv(file.path(out, "gene_modules.csv"))
rows <- list()
for (m in setdiff(unique(primary$module), "grey")) {
  a <- primary$gene_name[primary$module == m]
  for (s in setdiff(unique(sens$module), "grey")) {
    b <- sens$gene_name[sens$module == s]
    rows[[length(rows)+1]] <- data.frame(primary_module = m, sensitivity_module = s,
                                       primary_size = length(a), sensitivity_size = length(b),
                                       overlap = length(intersect(a,b)),
                                       jaccard = length(intersect(a,b))/length(union(a,b)),
                                       fraction_primary_retained = length(intersect(a,b))/length(a))
  }
}
tab <- bind_rows(rows)
write.csv(tab, file.path(out,"QC_sensitivity_all_module_overlaps.csv"), row.names=FALSE)
best <- tab %>% group_by(primary_module) %>% slice_max(jaccard, n=1, with_ties=FALSE) %>% ungroup()
write.csv(best, file.path(out,"QC_sensitivity_best_module_matches.csv"), row.names=FALSE)
print(best)
jsonlite::write_json(list(n_metacells=nrow(GetDatExpr(obj)), n_genes=ncol(GetDatExpr(obj)),
                         soft_power=beta, purpose="C5/C8 exclusion sensitivity at fixed genes and network parameters; overlap is descriptive, not formal preservation testing"),
                    file.path(out,"QC_sensitivity_metadata.json"), pretty=TRUE, auto_unbox=TRUE)
