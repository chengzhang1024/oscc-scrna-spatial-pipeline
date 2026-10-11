"""GO BP and KEGG over-representation analysis for Figure 1G epithelial DEGs."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
RESULT = ROOT / "results/paper9"
GENESET_DIR = RESULT / "gene_sets"
USER_CACHE = Path.home() / ".cache/gseapy"

COMPARISONS = {
    "OLK vs Normal": RESULT / "Fig1G_epithelial_celllevel_OLK_vs_Normal_display.csv.gz",
    "OSCC vs Normal": RESULT / "Fig1G_epithelial_celllevel_OSCC_vs_Normal_display.csv.gz",
}
LIBRARIES = {
    "GO_BP_2023": "GO_Biological_Process_2023",
    "KEGG_2021_Human": "KEGG_2021_Human",
}
FDR_THRESHOLD = 0.05
LOG2FC_THRESHOLD = 1.0
MIN_PATHWAY_SIZE = 5
MAX_PATHWAY_SIZE = 500
MIN_OVERLAP = 2


def read_gmt(path: Path) -> dict[str, set[str]]:
    output: dict[str, set[str]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                output[fields[0]] = {gene.upper() for gene in fields[2:] if gene}
    return output


def write_gmt(path: Path, gene_sets: dict[str, list[str] | set[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for term in sorted(gene_sets):
            genes = sorted({gene.upper() for gene in gene_sets[term] if gene})
            handle.write("\t".join([term, "Enrichr", *genes]) + "\n")


def load_library(short_name: str, enrichr_name: str) -> dict[str, set[str]]:
    project_cache = GENESET_DIR / f"Enrichr.{enrichr_name}.gmt"
    user_cache = USER_CACHE / f"Enrichr.{enrichr_name}.gmt"
    if project_cache.exists():
        path = project_cache
    elif user_cache.exists():
        path = user_cache
    else:
        print(f"Downloading {enrichr_name} from Enrichr...", flush=True)
        query = urlencode({"mode": "text", "libraryName": enrichr_name})
        url = f"https://maayanlab.cloud/Enrichr/geneSetLibrary?{query}"
        with urlopen(url, timeout=60) as response:
            text = response.read().decode("utf-8")
        downloaded = {}
        for line in text.splitlines():
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 3:
                downloaded[fields[0]] = {gene for gene in fields[2:] if gene}
        if not downloaded:
            raise RuntimeError(f"No gene sets returned for {enrichr_name}")
        write_gmt(project_cache, downloaded)
        path = project_cache
    gene_sets = read_gmt(path)
    print(f"Loaded {len(gene_sets):,} {short_name} terms from {path}", flush=True)
    return gene_sets


def enrich_one(
    comparison: str,
    direction: str,
    genes: set[str],
    universe: set[str],
    database: str,
    gene_sets: dict[str, set[str]],
) -> pd.DataFrame:
    query = genes & universe
    rows: list[dict[str, object]] = []
    for term, members in gene_sets.items():
        pathway = members & universe
        if not MIN_PATHWAY_SIZE <= len(pathway) <= MAX_PATHWAY_SIZE:
            continue
        overlap = query & pathway
        if len(overlap) < MIN_OVERLAP:
            continue
        a = len(overlap)
        b = len(query) - a
        c = len(pathway) - a
        d = len(universe) - a - b - c
        if min(a, b, c, d) < 0:
            continue
        odds_ratio, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append(
            {
                "comparison": comparison,
                "direction": direction,
                "database": database,
                "term": term,
                "overlap_count": a,
                "query_size": len(query),
                "pathway_size": len(pathway),
                "universe_size": len(universe),
                "gene_ratio": a / max(len(query), 1),
                "background_ratio": len(pathway) / max(len(universe), 1),
                "odds_ratio": odds_ratio,
                "pvalue": pvalue,
                "overlap_genes": ";".join(sorted(overlap)),
                "deg_threshold": f"FDR<{FDR_THRESHOLD}; |log2FC|>={LOG2FC_THRESHOLD}",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["fdr"] = multipletests(result["pvalue"], method="fdr_bh")[1]
    result["minus_log10_fdr"] = -np.log10(result["fdr"].clip(lower=np.finfo(float).tiny))
    return result.sort_values(["fdr", "pvalue", "term"]).reset_index(drop=True)


def main() -> None:
    RESULT.mkdir(parents=True, exist_ok=True)
    frames = {comparison: pd.read_csv(path) for comparison, path in COMPARISONS.items()}
    common_universe = set.intersection(
        *[{str(gene).upper() for gene in frame["gene"].dropna()} for frame in frames.values()]
    )
    libraries = {
        short_name: load_library(short_name, enrichr_name)
        for short_name, enrichr_name in LIBRARIES.items()
    }

    gene_list_rows: list[dict[str, object]] = []
    enrichment_frames: list[pd.DataFrame] = []
    for comparison, frame in frames.items():
        frame = frame.copy()
        frame["gene_upper"] = frame["gene"].astype(str).str.upper()
        selections = {
            "Up": (frame["fdr"] < FDR_THRESHOLD) & (frame["log2_cpm_diff"] >= LOG2FC_THRESHOLD),
            "Down": (frame["fdr"] < FDR_THRESHOLD) & (frame["log2_cpm_diff"] <= -LOG2FC_THRESHOLD),
        }
        for direction, mask in selections.items():
            selected = frame.loc[mask & frame["gene_upper"].isin(common_universe)].copy()
            selected_genes = set(selected["gene_upper"])
            print(f"{comparison} {direction}: {len(selected_genes):,} DEGs", flush=True)
            for row in selected.sort_values("fdr").itertuples(index=False):
                gene_list_rows.append(
                    {
                        "comparison": comparison,
                        "direction": direction,
                        "gene": row.gene,
                        "log2FC": row.log2_cpm_diff,
                        "pvalue": row.pvalue,
                        "fdr": row.fdr,
                    }
                )
            for database, gene_sets in libraries.items():
                enriched = enrich_one(
                    comparison,
                    direction,
                    selected_genes,
                    common_universe,
                    database,
                    gene_sets,
                )
                if not enriched.empty:
                    enrichment_frames.append(enriched)

    gene_lists = pd.DataFrame(gene_list_rows)
    gene_lists.to_csv(RESULT / "Fig1G_epithelial_enrichment_DEG_lists.csv", index=False, encoding="utf-8-sig")
    all_results = pd.concat(enrichment_frames, ignore_index=True)
    all_results = all_results.sort_values(["comparison", "direction", "database", "fdr", "pvalue"])
    significant = all_results[all_results["fdr"] < 0.05].copy()

    all_results.to_csv(
        RESULT / "Fig1G_epithelial_GO_KEGG_enrichment_all.csv", index=False, encoding="utf-8-sig"
    )
    significant.to_csv(
        RESULT / "Fig1G_epithelial_GO_KEGG_enriched_pathways.csv", index=False, encoding="utf-8-sig"
    )
    for database, filename in {
        "GO_BP_2023": "Fig1G_epithelial_GO_BP_enriched_pathways.csv",
        "KEGG_2021_Human": "Fig1G_epithelial_KEGG_enriched_pathways.csv",
    }.items():
        significant[significant["database"] == database].to_csv(
            RESULT / filename, index=False, encoding="utf-8-sig"
        )

    summary = (
        significant.groupby(["comparison", "direction", "database"], observed=True)
        .size()
        .rename("n_fdr_significant_pathways")
        .reset_index()
    )
    summary.to_csv(RESULT / "Fig1G_epithelial_enrichment_summary.csv", index=False, encoding="utf-8-sig")
    print(summary.to_string(index=False), flush=True)
    print(f"Common background: {len(common_universe):,} genes", flush=True)


if __name__ == "__main__":
    main()
