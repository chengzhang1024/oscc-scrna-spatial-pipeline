"""Literature-v3 sample-level pseudobulk pathway analysis for Figure 2H.

The primary comparison is OSCC versus Normal within GSE181919. Raw counts are
aggregated per sample and unified major cell type, normalized to log2 CPM, and
tested at the sample level to avoid single-cell pseudoreplication.
"""

from pathlib import Path
import warnings

import gseapy as gp
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact, mannwhitneyu
from statsmodels.stats.multitest import multipletests


ROOT = Path(__file__).resolve().parent
H5AD = ROOT / "results/integrated_3ds_annotated_literature_v3.h5ad"
RESULT = ROOT / "results"
FIGURE = ROOT / "figures"
PDF_FIGURE = FIGURE / "pdf"

CELL_TYPES = [
    "Epithelial cells", "T/NK cells", "B/Plasma cells", "Myeloid cells",
    "Fibroblasts", "Endothelial cells", "Myocytes", "Mast cells",
]

METABOLIC_KEYWORDS = (
    "metabolism", "glycolysis", "gluconeogenesis", "citrate cycle", "tca cycle",
    "pentose phosphate", "oxidative phosphorylation", "fatty acid", "amino acid",
    "biosynthesis", "degradation", "cholesterol", "bile secretion", "purine",
    "pyrimidine", "sphingolipid", "glycerolipid", "glycerophospholipid", "retinol",
    "cytochrome p450", "carbon fixation", "carbon metabolism", "one carbon pool",
    "pyruvate", "propanoate", "butanoate", "galactose", "fructose", "mannose",
    "starch and sucrose", "glycan", "glycosylation", "lysosome", "peroxisome",
)


def decode(values):
    return np.asarray([
        value.decode("utf-8") if isinstance(value, bytes) else str(value)
        for value in values
    ], dtype=object)


def read_categorical(group, name):
    item = group[name]
    categories = decode(item["categories"][:])
    codes = item["codes"][:]
    output = np.full(len(codes), "", dtype=object)
    valid = codes >= 0
    output[valid] = categories[codes[valid]]
    return output


def aggregate_pseudobulk(chunk_size=5000):
    print("Reading metadata and aggregating raw counts...", flush=True)
    with h5py.File(H5AD, "r") as h5:
        obs = h5["obs"]
        dataset = read_categorical(obs, "dataset")
        disease = read_categorical(obs, "disease_status")
        patient = read_categorical(obs, "patient_id")
        cell_type = read_categorical(obs, "cell_type")
        genes = decode(h5["var"]["_index"][:])

        selected = (
            (dataset == "A4_GSE181919")
            & np.isin(disease, ["Normal", "OSCC"])
            & np.isin(cell_type, CELL_TYPES)
        )
        group_keys = np.asarray([
            f"{patient[index]}|{disease[index]}|{cell_type[index]}"
            for index in np.flatnonzero(selected)
        ], dtype=object)
        unique_keys, local_codes = np.unique(group_keys, return_inverse=True)
        global_codes = np.full(len(dataset), -1, dtype=np.int32)
        global_codes[np.flatnonzero(selected)] = local_codes

        counts = np.zeros((len(unique_keys), len(genes)), dtype=np.float64)
        n_cells = np.zeros(len(unique_keys), dtype=np.int32)
        x_group = h5["layers"]["counts"]
        indptr = x_group["indptr"][:]

        for start in range(0, len(dataset), chunk_size):
            end = min(start + chunk_size, len(dataset))
            codes = global_codes[start:end]
            if not (codes >= 0).any():
                continue
            data_start = int(indptr[start])
            data_end = int(indptr[end])
            chunk = sparse.csr_matrix(
                (
                    x_group["data"][data_start:data_end],
                    x_group["indices"][data_start:data_end],
                    indptr[start:end + 1] - data_start,
                ),
                shape=(end - start, len(genes)),
            )
            for code in np.unique(codes[codes >= 0]):
                mask = codes == code
                counts[code] += np.asarray(chunk[mask].sum(axis=0)).ravel()
                n_cells[code] += int(mask.sum())
            if end % 50000 == 0 or end == len(dataset):
                print(f"  processed {end:,}/{len(dataset):,} cells", flush=True)

    metadata = pd.DataFrame(
        [key.split("|", 2) for key in unique_keys],
        columns=["patient_id", "disease_status", "cell_type"],
    )
    metadata["n_cells"] = n_cells
    metadata["library_size"] = counts.sum(axis=1)
    metadata.to_csv(RESULT / "fig2h_literature_v3_pseudobulk_samples.csv", index=False, encoding="utf-8-sig")
    return counts, metadata, genes


def pseudobulk_de(counts, metadata, genes, min_cells=10, min_samples=3):
    rows = []
    summary_rows = []
    degs = {}
    valid_cell_types = []

    for cell_type in CELL_TYPES:
        mask = (metadata["cell_type"] == cell_type) & (metadata["n_cells"] >= min_cells)
        ct_meta = metadata.loc[mask].reset_index(drop=True)
        ct_counts = counts[mask.to_numpy()]
        normal = ct_meta["disease_status"].to_numpy() == "Normal"
        oscc = ct_meta["disease_status"].to_numpy() == "OSCC"
        if normal.sum() < min_samples or oscc.sum() < min_samples:
            print(
                f"  skip {cell_type}: {normal.sum()} Normal and {oscc.sum()} OSCC samples",
                flush=True,
            )
            continue

        library = ct_counts.sum(axis=1, keepdims=True)
        log_cpm = np.log2((ct_counts / np.maximum(library, 1)) * 1_000_000 + 1)
        detected = (log_cpm > 1).sum(axis=0) >= max(2, int(np.ceil(0.2 * len(ct_meta))))
        tested_genes = genes[detected]
        tested = log_cpm[:, detected]

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, pvalue = mannwhitneyu(
                tested[oscc], tested[normal], alternative="two-sided",
                axis=0, method="asymptotic",
            )
        log_fc = tested[oscc].mean(axis=0) - tested[normal].mean(axis=0)
        fdr = multipletests(pvalue, method="fdr_bh")[1]
        frame = pd.DataFrame({
            "cell_type": cell_type,
            "gene": tested_genes,
            "log2_cpm_diff": log_fc,
            "statistic": statistic,
            "pvalue": pvalue,
            "fdr": fdr,
            "n_normal_samples": int(normal.sum()),
            "n_oscc_samples": int(oscc.sum()),
        }).sort_values("pvalue")
        rows.append(frame)

        # FDR is used for the main gene lists. A nominally supported effect-size
        # list is retained only when sample-level power yields too few genes for ORA.
        up = frame[(frame["fdr"] < 0.10) & (frame["log2_cpm_diff"] > 0.5)]["gene"].tolist()
        down = frame[(frame["fdr"] < 0.10) & (frame["log2_cpm_diff"] < -0.5)]["gene"].tolist()
        up_selection = "FDR<0.10"
        down_selection = "FDR<0.10"
        if len(up) < 20:
            up = frame[(frame["pvalue"] < 0.05) & (frame["log2_cpm_diff"] > 0.5)]["gene"].tolist()
            up_selection = "nominal P<0.05 (low-power fallback)"
        if len(down) < 20:
            down = frame[(frame["pvalue"] < 0.05) & (frame["log2_cpm_diff"] < -0.5)]["gene"].tolist()
            down_selection = "nominal P<0.05 (low-power fallback)"
        degs[cell_type] = {"up": up, "down": down, "universe": set(tested_genes)}
        valid_cell_types.append(cell_type)
        summary_rows.append({
            "cell_type": cell_type,
            "n_normal_samples": int(normal.sum()),
            "n_oscc_samples": int(oscc.sum()),
            "minimum_cells_per_sample": min_cells,
            "n_up_genes": len(up),
            "n_down_genes": len(down),
            "up_selection": up_selection,
            "down_selection": down_selection,
        })
        print(
            f"  {cell_type}: {normal.sum()} vs {oscc.sum()} samples; "
            f"{len(up)} up [{up_selection}], {len(down)} down [{down_selection}]",
            flush=True,
        )

    de = pd.concat(rows, ignore_index=True)
    de.to_csv(RESULT / "fig2h_literature_v3_pseudobulk_de.csv.gz", index=False, compression="gzip")
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(
        RESULT / "fig2h_literature_v3_de_summary.csv", index=False, encoding="utf-8-sig"
    )
    return degs, valid_cell_types, summary


def load_gene_sets(library, metabolic_only=False):
    gene_sets = gp.get_library(name=library, organism="Human")
    output = {term: set(genes) for term, genes in gene_sets.items()}
    if metabolic_only:
        output = {
            term: genes for term, genes in output.items()
            if any(keyword in term.lower() for keyword in METABOLIC_KEYWORDS)
        }
    print(f"  loaded {len(output)} pathways from {library}", flush=True)
    return output


def enrich_one(gene_list, universe, gene_sets):
    query = set(gene_list) & universe
    rows = []
    if len(query) < 3:
        return pd.DataFrame()
    for term, members in gene_sets.items():
        pathway = members & universe
        overlap = query & pathway
        if len(overlap) < 3 or len(pathway) < 5:
            continue
        a = len(overlap)
        b = len(query) - a
        c = len(pathway) - a
        d = len(universe) - a - b - c
        if d < 0:
            continue
        _, pvalue = fisher_exact([[a, b], [c, d]], alternative="greater")
        rows.append({
            "term": term, "pvalue": pvalue, "overlap": a,
            "query_size": len(query), "pathway_size": len(pathway),
            "genes": ";".join(sorted(overlap)),
        })
    if not rows:
        return pd.DataFrame()
    result = pd.DataFrame(rows)
    result["fdr"] = multipletests(result["pvalue"], method="fdr_bh")[1]
    return result.sort_values("pvalue")


def enrichment_matrix(degs, cell_types, gene_sets, library_label, top_n=30):
    rows = []
    for cell_type in cell_types:
        for direction in ("up", "down"):
            enriched = enrich_one(degs[cell_type][direction], degs[cell_type]["universe"], gene_sets)
            if enriched.empty:
                continue
            enriched["cell_type"] = cell_type
            enriched["direction"] = direction
            rows.append(enriched)
    if not rows:
        raise RuntimeError(f"No enrichment results for {library_label}")
    long = pd.concat(rows, ignore_index=True)
    long.to_csv(
        RESULT / f"fig2h_literature_v3_{library_label}_enrichment.csv", index=False, encoding="utf-8-sig"
    )

    significant = long[long["fdr"] < 0.10].copy()
    if significant.empty:
        significant = long[long["pvalue"] < 0.05].copy()
    ranking = significant.groupby("term").agg(
        n_cell_types=("cell_type", "nunique"), min_fdr=("fdr", "min")
    ).sort_values(["n_cell_types", "min_fdr"], ascending=[False, True])
    terms = ranking.head(top_n).index.tolist()
    if not terms:
        terms = long.sort_values("pvalue")["term"].drop_duplicates().head(top_n).tolist()

    matrix = pd.DataFrame(0.0, index=terms, columns=cell_types)
    fdr_matrix = pd.DataFrame(1.0, index=terms, columns=cell_types)
    for _, row in long[long["term"].isin(terms)].iterrows():
        value = -np.log10(max(row["fdr"], 1e-10))
        if row["direction"] == "down":
            value *= -1
        current = matrix.loc[row["term"], row["cell_type"]]
        if abs(value) > abs(current):
            matrix.loc[row["term"], row["cell_type"]] = value
            fdr_matrix.loc[row["term"], row["cell_type"]] = row["fdr"]
    matrix.to_csv(RESULT / f"fig2h_literature_v3_{library_label}_matrix.csv", encoding="utf-8-sig")
    return matrix, fdr_matrix


def clean_term(term):
    for prefix in ("HALLMARK_", "KEGG_"):
        term = term.replace(prefix, "")
    return term.replace("_", " ").title()


def plot_heatmap(matrix, fdr, title, output_name, low_power_types=None):
    low_power_types = set(low_power_types or [])
    height = max(7.5, 0.27 * len(matrix) + 2)
    width = max(10.5, 1.05 * len(matrix.columns) + 3)
    fig, ax = plt.subplots(figsize=(width, height))
    values = np.clip(matrix.to_numpy(), -6, 6)
    cmap = LinearSegmentedColormap.from_list("direction", ["#3B78B5", "#F7F7F7", "#D95F59"])
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-6, vmax=6, interpolation="nearest")
    ax.set_xticks(np.arange(len(matrix.columns)))
    ax.set_xticklabels([
        item.replace(" ", "\n") + ("\n(low power)" if item in low_power_types else "")
        for item in matrix.columns
    ], fontsize=8)
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels([clean_term(item) for item in matrix.index], fontsize=7)
    for row in range(len(matrix.index)):
        for column in range(len(matrix.columns)):
            adjusted = fdr.iloc[row, column]
            marker = "**" if adjusted < 0.05 else "*" if adjusted < 0.10 else ""
            if marker:
                ax.text(column, row, marker, ha="center", va="center", color="white", fontsize=7, weight="bold")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Signed -log10(FDR): OSCC vs Normal", fontsize=8)
    ax.set_title(title, fontsize=11, weight="bold")
    fig.tight_layout()
    for extension in ("png", "pdf"):
        output_dir = PDF_FIGURE if extension == "pdf" else FIGURE
        output_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_dir / f"{output_name}.{extension}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    counts, metadata, genes = aggregate_pseudobulk()
    degs, cell_types, summary = pseudobulk_de(counts, metadata, genes)
    low_power_types = summary.loc[
        summary["up_selection"].str.startswith("nominal")
        | summary["down_selection"].str.startswith("nominal"),
        "cell_type",
    ].tolist()

    print("KEGG metabolic enrichment...", flush=True)
    kegg = load_gene_sets("KEGG_2021_Human", metabolic_only=True)
    kegg_matrix, kegg_fdr = enrichment_matrix(degs, cell_types, kegg, "KEGG_metabolic", top_n=30)
    plot_heatmap(
        kegg_matrix, kegg_fdr,
        "H1  Metabolic pathway reprogramming (GSE181919 pseudobulk)",
        "Fig2H1_final_KEGG_metabolic_pseudobulk",
        low_power_types,
    )

    print("Hallmark enrichment...", flush=True)
    hallmark = load_gene_sets("MSigDB_Hallmark_2020")
    hallmark_matrix, hallmark_fdr = enrichment_matrix(degs, cell_types, hallmark, "Hallmark", top_n=30)
    plot_heatmap(
        hallmark_matrix, hallmark_fdr,
        "H2  Hallmark pathway reprogramming (GSE181919 pseudobulk)",
        "Fig2H2_final_Hallmark_pseudobulk",
        low_power_types,
    )
    print("Figure 2H literature-v3 complete.", flush=True)


if __name__ == "__main__":
    main()
