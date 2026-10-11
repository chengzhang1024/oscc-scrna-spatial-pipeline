"""Build epithelial Figure 3G in the reference LandSCENT layout.

The analysis follows the LandSCENT definitions used by the reference paper:
single-cell signaling entropy rate (SR) is calculated on the official
net17Jan16 human PPI, potency states are inferred from logit(SR), and a
30-neighbor diffusion map is rooted at the cell with maximum SR.  The left
side reproduces the high/low-potency density landscapes and the right side
shows the three-dimensional diffusion map colored by SR.
"""

from __future__ import annotations

import gzip
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CACHE_ROOT = ROOT / ".cache"
TASK_TEMP = CACHE_ROOT / "codex_tmp"
NUMBA_CACHE = CACHE_ROOT / "numba_cache"
LANDSCENT_CACHE = CACHE_ROOT / "landscent"
for directory in (TASK_TEMP, NUMBA_CACHE, LANDSCENT_CACHE):
    directory.mkdir(parents=True, exist_ok=True)
os.environ["TEMP"] = str(TASK_TEMP)
os.environ["TMP"] = str(TASK_TEMP)
os.environ["NUMBA_CACHE_DIR"] = str(NUMBA_CACHE)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")

import anndata as ad
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.csgraph import connected_components, dijkstra
from scipy.sparse.linalg import eigsh
from scipy.stats import gaussian_kde, kruskal, spearmanr
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors


matplotlib.rcParams.update(
    {
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "axes.linewidth": 0.9,
    }
)

H5AD = ROOT / "results/integrated_3ds_annotated_paper9.h5ad"
FIG3A_CELLS = (
    ROOT / "results/paper9/figure3_epithelial/Fig3A_epithelial_cells_and_umap.csv.gz"
)
FIG3A_HVGS = ROOT / "results/paper9/figure3_epithelial/Fig3A_epithelial_hvg_genes.csv"
LANDSCENT_RDATA = (
    ROOT / "external/LandSCENT_official/data/net17Jan16.m.RData"
)
LANDSCENT_REPOSITORY = ROOT / "external/LandSCENT_official"
NCBI_GENE_INFO = (
    ROOT / "data/reference_annotation/ncbi_gene_info/Homo_sapiens.gene_info.gz"
)
RDATA_VENDOR = ROOT / ".vendor/rdata"
PPI_CACHE = LANDSCENT_CACHE / "net17Jan16_sparse.npz"
PPI_GENE_CACHE = LANDSCENT_CACHE / "net17Jan16_genes.txt.gz"
RESULT_DIR = ROOT / "results/paper9/figure3_epithelial"
FIGURE_DIR = ROOT / "figures/paper9/figure3_epithelial"
OUTPUT_STEM = FIGURE_DIR / "Fig3G_epithelial_LandSCENT_trajectory_reference_style"

SEED = 20260902
K_NEIGHBORS = 30
N_DIFFUSION_COMPONENTS = 3
N_PCA_COMPONENTS = 30
LANDSCENT_MEAN_GAP = 1.0
LANDSCENT_SD_GAP = 1.0
PRIMARY_MAX_POTENCY_STATES = 2
DEFAULT_MAX_POTENCY_STATES = 5
EXPRESSION_OFFSET = 1.1
BASELINE = float(np.log2(EXPRESSION_OFFSET))
CLUSTERS = [f"C{i}" for i in range(9)]
QC_CLUSTERS = {"C5", "C8"}

SR_CMAP = LinearSegmentedColormap.from_list(
    "landscent_sr", ["#ADD8E6", "#66C2E8", "#3156B5", "#000080"]
)
HIGH_CMAP = LinearSegmentedColormap.from_list(
    "landscent_high", ["#D9D9D9", "#FFD92F", "#FF8C00", "#E31A1C"]
)
LOW_CMAP = LinearSegmentedColormap.from_list(
    "landscent_low", ["#D9D9D9", "#4DAF4A", "#1857C9", "#071B8C"]
)
ALL_CMAP = LinearSegmentedColormap.from_list(
    "landscent_all", ["#FFFFFF", "#F0A3F2", "#FF00D4", "#E31A1C"]
)


def technical_gene(gene: str) -> bool:
    """Only used for the documented diffusion-map fallback."""
    return (
        gene.startswith("MT-")
        or gene.startswith("RPL")
        or gene.startswith("RPS")
        or gene.startswith("IGH")
        or gene.startswith("IGK")
        or gene.startswith("IGL")
        or gene.startswith("TRAV")
        or gene.startswith("TRBV")
    )


def load_official_ppi() -> tuple[sparse.csr_matrix, np.ndarray]:
    """Load the official LandSCENT net17Jan16 adjacency, caching it sparsely."""
    if PPI_CACHE.exists() and PPI_GENE_CACHE.exists():
        adjacency = sparse.load_npz(PPI_CACHE).tocsr().astype(np.float64)
        with gzip.open(PPI_GENE_CACHE, "rt", encoding="utf-8") as handle:
            genes = np.asarray([line.strip() for line in handle if line.strip()], dtype=str)
        if adjacency.shape != (len(genes), len(genes)):
            raise RuntimeError("Cached LandSCENT PPI dimensions do not match its gene list")
        return adjacency, genes

    if not LANDSCENT_RDATA.exists():
        raise FileNotFoundError(
            "Official LandSCENT net17Jan16.m.RData is missing; clone "
            "https://github.com/ChenWeiyan/LandSCENT first"
        )
    if str(RDATA_VENDOR) not in sys.path:
        sys.path.append(str(RDATA_VENDOR))
    try:
        import rdata
    except ImportError as exc:
        raise RuntimeError(
            "The local rdata reader is missing; install rdata under .vendor/rdata"
        ) from exc

    loaded = rdata.read_rda(LANDSCENT_RDATA)
    network = loaded["net17Jan16.m"]
    genes = np.asarray(network.coords["dim_0"].to_numpy(), dtype=str)
    adjacency = sparse.csr_matrix(np.asarray(network.data), dtype=np.float64)
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    if (adjacency != adjacency.T).nnz:
        raise RuntimeError("Official LandSCENT net17Jan16 network is not symmetric")
    sparse.save_npz(PPI_CACHE, adjacency, compressed=True)
    with gzip.open(PPI_GENE_CACHE, "wt", encoding="utf-8", newline="\n") as handle:
        handle.write("\n".join(genes.tolist()) + "\n")
    return adjacency, genes


def load_cells_and_expression() -> tuple[pd.DataFrame, sparse.csr_matrix, np.ndarray]:
    cells = pd.read_csv(FIG3A_CELLS)
    if len(cells) != 5609 or set(cells["epithelial_cluster"].astype(str)) != set(CLUSTERS):
        raise RuntimeError("Figure 3A cell table is not the fixed 5,609-cell C0-C8 input")

    backed = ad.read_h5ad(H5AD, backed="r")
    lookup = pd.Series(np.arange(backed.n_obs), index=backed.obs_names.astype(str))
    rows = lookup.reindex(cells["cell_id"].astype(str)).to_numpy()
    if np.isnan(rows).any():
        backed.file.close()
        raise RuntimeError("Some Figure 3A cells are absent from the source object")
    rows = rows.astype(int)
    source_obs = backed.obs.iloc[rows]
    for column in ("n_genes_by_counts", "total_counts", "pct_counts_mt"):
        if column in source_obs:
            cells[column] = pd.to_numeric(source_obs[column].to_numpy(), errors="coerce")

    matrix = backed.X[rows, :]
    if hasattr(matrix, "to_memory"):
        matrix = matrix.to_memory()
    matrix = sparse.csr_matrix(matrix, dtype=np.float64)
    genes = backed.var_names.astype(str).to_numpy()
    backed.file.close()

    # The object stores ln(1 + normalized abundance). LandSCENT recommends
    # log2(normalized abundance + 1.1), which keeps every entry positive.
    delta = matrix.copy()
    delta.data = np.log2(np.expm1(delta.data) + EXPRESSION_OFFSET) - BASELINE
    delta.eliminate_zeros()
    return cells, delta, genes


def load_ncbi_symbol_map() -> pd.DataFrame:
    if not NCBI_GENE_INFO.exists():
        raise FileNotFoundError("NCBI Homo_sapiens.gene_info.gz is missing")
    mapping = pd.read_csv(
        NCBI_GENE_INFO,
        sep="\t",
        compression="gzip",
        dtype=str,
        usecols=["#tax_id", "GeneID", "Symbol", "Synonyms"],
        low_memory=False,
    )
    mapping = mapping[mapping["#tax_id"].eq("9606")].copy()
    mapping = mapping.drop_duplicates("Symbol", keep="first")
    return mapping


def integrate_expression_and_ppi(
    adjacency: sparse.csr_matrix,
    ppi_genes: np.ndarray,
    expression_genes: np.ndarray,
) -> tuple[sparse.csr_matrix, np.ndarray, np.ndarray, pd.DataFrame]:
    ncbi = load_ncbi_symbol_map()
    symbol_to_entrez = pd.Series(ncbi["GeneID"].to_numpy(), index=ncbi["Symbol"]).to_dict()
    ppi_lookup = {gene: index for index, gene in enumerate(ppi_genes)}

    candidates = []
    for column, symbol in enumerate(expression_genes):
        entrez = symbol_to_entrez.get(str(symbol))
        if entrez is not None and entrez in ppi_lookup:
            candidates.append(
                {
                    "symbol": str(symbol),
                    "entrez_id": str(entrez),
                    "expression_column": int(column),
                    "ppi_index": int(ppi_lookup[entrez]),
                }
            )
    mapped = pd.DataFrame(candidates)
    if mapped.empty:
        raise RuntimeError("No expression genes map to the official LandSCENT PPI")
    mapped = mapped.sort_values(
        ["ppi_index", "symbol"], kind="mergesort"
    ).drop_duplicates("entrez_id", keep="first")

    sub = adjacency[mapped["ppi_index"].to_numpy()][:, mapped["ppi_index"].to_numpy()]
    n_components, labels = connected_components(sub, directed=False)
    component_sizes = np.bincount(labels)
    largest = int(np.argmax(component_sizes))
    keep = labels == largest
    mapped = mapped.loc[keep].reset_index(drop=True)
    sub = sub[keep][:, keep].tocsr().astype(np.float64)
    sub.data[:] = 1.0
    sub.eliminate_zeros()
    if n_components > 1 and sub.shape[0] < 1000:
        raise RuntimeError("Mapped LandSCENT PPI largest component is unexpectedly small")
    return (
        sub,
        mapped["expression_column"].to_numpy(dtype=int),
        mapped["entrez_id"].to_numpy(dtype=str),
        mapped,
    )


def calculate_signaling_entropy(
    delta: sparse.csr_matrix,
    adjacency: sparse.csr_matrix,
    batch_size: int = 128,
) -> tuple[np.ndarray, float, float]:
    if delta.shape[1] != adjacency.shape[0]:
        raise RuntimeError("Expression/PPI dimensions differ during SR calculation")
    largest_eigenvalue = float(
        eigsh(adjacency, k=1, which="LA", return_eigenvectors=False, tol=1e-6)[0]
    )
    maximum_entropy_rate = float(np.log(largest_eigenvalue))
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    if np.any(degree <= 0) or maximum_entropy_rate <= 0:
        raise RuntimeError("LandSCENT PPI is not a valid connected entropy scaffold")

    sr = np.empty(delta.shape[0], dtype=np.float64)
    baseline_neighbor_sum = BASELINE * degree
    for start in range(0, delta.shape[0], batch_size):
        stop = min(start + batch_size, delta.shape[0])
        block = delta[start:stop]
        expression = block.toarray()
        expression += BASELINE
        neighbor_sum = (block @ adjacency).toarray()
        neighbor_sum += baseline_neighbor_sum[None, :]
        flux = expression * neighbor_sum
        denominator = flux.sum(axis=1)
        numerator = (
            flux * (np.log(neighbor_sum) - np.log(expression))
        ).sum(axis=1)
        sr[start:stop] = numerator / denominator / maximum_entropy_rate
        print(f"  SR cells {start + 1:,}-{stop:,}/{delta.shape[0]:,}", flush=True)
    if not np.isfinite(sr).all() or np.any((sr <= 0) | (sr >= 1)):
        raise RuntimeError("Calculated SR values fall outside the expected open interval (0,1)")
    return sr, largest_eigenvalue, maximum_entropy_rate


def fit_potency_states(
    sr: np.ndarray,
    max_states: int,
    analysis: str,
) -> tuple[np.ndarray, pd.DataFrame, GaussianMixture]:
    logit = np.log2(sr / (1.0 - sr)).reshape(-1, 1)
    rows = []
    models: dict[int, GaussianMixture] = {}
    for n_states in range(1, max_states + 1):
        model = GaussianMixture(
            n_components=n_states,
            covariance_type="full",
            n_init=20,
            reg_covar=1e-7,
            random_state=SEED,
        ).fit(logit)
        models[n_states] = model
        rows.append(
            {
                "analysis": analysis,
                "n_states": n_states,
                "bic": float(model.bic(logit)),
                "aic": float(model.aic(logit)),
                "converged": bool(model.converged_),
            }
        )
    audit = pd.DataFrame(rows)
    selected_n = int(audit.loc[audit["bic"].idxmin(), "n_states"])
    model = models[selected_n]
    raw_labels = model.predict(logit)
    component_sr = 2.0 ** model.means_.ravel() / (1.0 + 2.0 ** model.means_.ravel())
    ordered_components = np.argsort(-component_sr)
    state_lookup = {int(component): rank + 1 for rank, component in enumerate(ordered_components)}
    states = np.asarray([state_lookup[int(label)] for label in raw_labels], dtype=int)
    audit["selected"] = audit["n_states"].eq(selected_n)
    audit["selected_component_mean_sr"] = ""
    means = ";".join(
        f"PS{rank + 1}:{component_sr[component]:.8f}"
        for rank, component in enumerate(ordered_components)
    )
    audit.loc[audit["selected"], "selected_component_mean_sr"] = means
    return states, audit, model


def select_diffusion_genes(
    delta_all: sparse.csr_matrix,
    genes: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame, str]:
    n_cells = delta_all.shape[0]
    delta_sum = np.asarray(delta_all.sum(axis=0)).ravel()
    delta_sq_sum = np.asarray(delta_all.power(2).sum(axis=0)).ravel()
    mean_delta = delta_sum / n_cells
    means = BASELINE + mean_delta
    second_moment = BASELINE**2 + 2 * BASELINE * mean_delta + delta_sq_sum / n_cells
    standard_deviations = np.sqrt(np.maximum(second_moment - means**2, 0.0))
    selected = (means > LANDSCENT_MEAN_GAP) & (standard_deviations > LANDSCENT_SD_GAP)
    selection_rule = "LandSCENT mean>1 and SD>1 on log2(normalized abundance+1.1)"

    if selected.sum() < 100:
        figure3a_hvgs = pd.read_csv(FIG3A_HVGS)["gene"].astype(str)
        allowed = set(figure3a_hvgs) & {
            str(gene) for gene in genes if not technical_gene(str(gene))
        }
        selected = np.asarray([str(gene) in allowed for gene in genes], dtype=bool)
        selection_rule = (
            "Fallback to Figure 3A patient-aware nontechnical HVGs because fewer "
            "than 100 genes passed the original LandSCENT mean/SD thresholds"
        )
    audit = pd.DataFrame(
        {
            "gene": genes,
            "mean_log2_offset_expression": means,
            "sd_log2_offset_expression": standard_deviations,
            "selected_for_diffusion_map": selected,
        }
    )
    return np.flatnonzero(selected), audit, selection_rule


def build_diffusion_map(
    delta_all: sparse.csr_matrix,
    selected_columns: np.ndarray,
) -> tuple[np.ndarray, sparse.csr_matrix, np.ndarray, np.ndarray]:
    expression = delta_all[:, selected_columns].toarray().astype(np.float32)
    n_components = min(N_PCA_COMPONENTS, expression.shape[0] - 1, expression.shape[1])
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        random_state=SEED,
    ).fit_transform(expression)
    del expression

    neighbors = NearestNeighbors(n_neighbors=K_NEIGHBORS + 1, metric="euclidean", n_jobs=-1)
    neighbors.fit(pca)
    distances, indices = neighbors.kneighbors(pca)
    distances = distances[:, 1:]
    indices = indices[:, 1:]
    sigma = np.maximum(distances[:, -1], np.finfo(float).eps)
    rows = np.repeat(np.arange(pca.shape[0]), K_NEIGHBORS)
    columns = indices.ravel()
    denominator = sigma[rows] * sigma[columns]
    weights = np.exp(-(distances.ravel() ** 2) / np.maximum(denominator, 1e-12))
    kernel = sparse.csr_matrix((weights, (rows, columns)), shape=(len(pca), len(pca)))
    kernel = kernel.maximum(kernel.T)
    kernel.setdiag(0)
    kernel.eliminate_zeros()

    # Density-normalized diffusion maps (alpha=1), represented by a symmetric
    # conjugate matrix for stable eigendecomposition.
    q = np.asarray(kernel.sum(axis=1)).ravel()
    q_inv = 1.0 / np.maximum(q, 1e-12)
    alpha_kernel = sparse.diags(q_inv) @ kernel @ sparse.diags(q_inv)
    row_sum = np.asarray(alpha_kernel.sum(axis=1)).ravel()
    d_inv_sqrt = 1.0 / np.sqrt(np.maximum(row_sum, 1e-12))
    symmetric = sparse.diags(d_inv_sqrt) @ alpha_kernel @ sparse.diags(d_inv_sqrt)
    eigenvalues, eigenvectors = eigsh(
        symmetric,
        k=N_DIFFUSION_COMPONENTS + 1,
        which="LA",
        tol=1e-6,
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    right_eigenvectors = d_inv_sqrt[:, None] * eigenvectors
    diffusion = right_eigenvectors[:, 1 : N_DIFFUSION_COMPONENTS + 1]
    diffusion *= eigenvalues[None, 1 : N_DIFFUSION_COMPONENTS + 1]

    # Weighted graph distances are retained for DPT-like endpoint selection.
    graph = sparse.csr_matrix(
        (distances.ravel(), (rows, columns)), shape=(len(pca), len(pca))
    )
    graph = graph.maximum(graph.T)
    graph.eliminate_zeros()
    return diffusion, graph, eigenvalues[1 : N_DIFFUSION_COMPONENTS + 1], pca


def orient_and_scale_diffusion(
    diffusion: np.ndarray,
    umap: np.ndarray,
) -> tuple[np.ndarray, list[int]]:
    oriented = diffusion.copy()
    signs = []
    targets = [umap[:, 0], umap[:, 1], umap[:, 0] + umap[:, 1]]
    for column in range(oriented.shape[1]):
        rho = spearmanr(oriented[:, column], targets[column]).statistic
        sign = 1 if not np.isfinite(rho) or rho >= 0 else -1
        oriented[:, column] *= sign
        signs.append(sign)
    centered = oriented - np.median(oriented, axis=0)
    scale = np.quantile(np.abs(centered), 0.98, axis=0)
    display = centered / np.maximum(scale, 1e-12)
    return display, signs


def select_root_and_terminals(
    sr: np.ndarray,
    graph: sparse.csr_matrix,
    diffusion_display: np.ndarray,
    clusters: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    root = int(np.argmax(sr))
    distances = np.asarray(dijkstra(graph, directed=False, indices=root)).ravel()
    finite = np.isfinite(distances)
    if finite.sum() != len(sr):
        raise RuntimeError("The 30-neighbor diffusion graph is disconnected")
    dpt = distances / max(float(distances.max()), 1e-12)

    local_maxima = []
    for cell in range(graph.shape[0]):
        start, stop = graph.indptr[cell], graph.indptr[cell + 1]
        neighborhood = graph.indices[start:stop]
        if len(neighborhood) and dpt[cell] >= dpt[neighborhood].max() and cell != root:
            local_maxima.append(cell)
    candidates = np.asarray(local_maxima, dtype=int)
    if len(candidates) < 2:
        candidates = np.argsort(dpt)[::-1][:100]

    first = int(candidates[np.argmax(dpt[candidates])])
    dc = diffusion_display
    separation = np.linalg.norm(dc[candidates] - dc[first], axis=1)
    separation /= max(float(separation.max()), 1e-12)
    score = 0.68 * dpt[candidates] + 0.32 * separation
    distinct = clusters[candidates] != clusters[first]
    eligible = (candidates != first) & distinct & (dpt[candidates] >= np.quantile(dpt, 0.85))
    if not eligible.any():
        eligible = candidates != first
    second_candidates = candidates[eligible]
    second = int(second_candidates[np.argmax(score[eligible])])
    terminals = np.asarray([first, second], dtype=int)
    terminals = terminals[np.argsort(-dpt[terminals])]
    return root, terminals, dpt


def density_grid(
    coordinates: np.ndarray,
    bounds: tuple[float, float, float, float],
    n_grid: int = 55,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xmin, xmax, ymin, ymax = bounds
    x = np.linspace(xmin, xmax, n_grid)
    y = np.linspace(ymin, ymax, n_grid)
    xx, yy = np.meshgrid(x, y)
    kde = gaussian_kde(coordinates.T, bw_method="scott")
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    zz *= len(coordinates)
    return xx, yy, zz


def style_landscape_axis(ax: plt.Axes) -> None:
    ax.view_init(elev=22, azim=-67)
    ax.set_xlabel("component 1", fontsize=8.0, labelpad=-2)
    ax.set_ylabel("component 2", fontsize=8.0, labelpad=-2)
    ax.set_zlabel("z", fontsize=8.0, labelpad=-4)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.grid(False)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor("black")
    ax.set_box_aspect((1.25, 1.0, 0.85))


def add_landscape(
    fig: plt.Figure,
    position: list[float],
    coordinates: np.ndarray,
    state_mask: np.ndarray,
    state_number: int,
    title: str,
    state_cmap: LinearSegmentedColormap,
    colorbar_x: float,
    colorbar_top_y: float,
    colorbar_bottom_y: float,
) -> None:
    ax = fig.add_axes(position, projection="3d")
    pad_x = 0.035 * np.ptp(coordinates[:, 0])
    pad_y = 0.035 * np.ptp(coordinates[:, 1])
    bounds = (
        float(coordinates[:, 0].min() - pad_x),
        float(coordinates[:, 0].max() + pad_x),
        float(coordinates[:, 1].min() - pad_y),
        float(coordinates[:, 1].max() + pad_y),
    )
    all_x, all_y, all_z = density_grid(coordinates, bounds, n_grid=55)
    state_x, state_y, state_z = density_grid(coordinates[state_mask], bounds, n_grid=70)
    state_z = state_z / max(float(state_z.max()), 1e-12)
    state_offset = 0.50 if state_number == 1 else 0.30
    state_surface = state_z + state_offset
    bottom = -1.0

    ax.plot_surface(
        state_x,
        state_y,
        state_surface,
        cmap=state_cmap,
        vmin=state_offset,
        vmax=state_offset + 1.0,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.98,
        rasterized=True,
    )
    ax.contourf(
        all_x,
        all_y,
        all_z,
        zdir="z",
        offset=bottom,
        levels=12,
        cmap=ALL_CMAP,
        alpha=0.98,
    )
    plane_x, plane_y = np.meshgrid(
        np.linspace(bounds[0], bounds[1], 2), np.linspace(bounds[2], bounds[3], 2)
    )
    ax.plot_surface(
        plane_x,
        plane_y,
        np.full_like(plane_x, state_offset - 0.05),
        color="#D9D9D9",
        alpha=0.35,
        linewidth=0,
        shade=False,
    )
    ax.set_xlim(bounds[0], bounds[1])
    ax.set_ylim(bounds[2], bounds[3])
    ax.set_zlim(bottom, 1.55)
    ax.set_title(title, fontsize=12.2, fontweight="bold", pad=2)
    style_landscape_axis(ax)

    top_cax = fig.add_axes([colorbar_x, colorbar_top_y, 0.010, 0.105])
    top_bar = fig.colorbar(
        ScalarMappable(
            norm=Normalize(state_offset, state_offset + 1.0), cmap=state_cmap
        ),
        cax=top_cax,
    )
    top_bar.set_ticks(np.linspace(state_offset, state_offset + 1.0, 5))
    top_bar.ax.tick_params(labelsize=6.0, width=0.6, length=2.5, pad=2)
    top_bar.outline.set_linewidth(0.7)
    fig.text(
        colorbar_x + 0.005,
        colorbar_top_y + 0.118,
        f"Cell Density\nPS{state_number}",
        fontsize=6.4,
        fontweight="bold",
        ha="center",
        va="bottom",
    )

    bottom_cax = fig.add_axes([colorbar_x, colorbar_bottom_y, 0.010, 0.105])
    all_norm = Normalize(float(all_z.min()), float(all_z.max()))
    bottom_bar = fig.colorbar(ScalarMappable(norm=all_norm, cmap=ALL_CMAP), cax=bottom_cax)
    bottom_bar.set_ticks(np.linspace(all_norm.vmin, all_norm.vmax, 4))
    bottom_bar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.0f"))
    bottom_bar.ax.tick_params(labelsize=5.8, width=0.6, length=2.5, pad=2)
    bottom_bar.outline.set_linewidth(0.7)
    fig.text(
        colorbar_x + 0.005,
        colorbar_bottom_y + 0.118,
        "Cell Density\nAll",
        fontsize=6.4,
        fontweight="bold",
        ha="center",
        va="bottom",
    )


def plot_figure(
    cells: pd.DataFrame,
    potency_states: np.ndarray,
    diffusion_display: np.ndarray,
    sr: np.ndarray,
    root: int,
    terminals: np.ndarray,
) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    coordinates = cells[["umap_1", "umap_2"]].to_numpy(dtype=float)
    n_states = int(potency_states.max())
    high_mask = potency_states == 1
    low_mask = potency_states == n_states
    if high_mask.sum() < 20 or low_mask.sum() < 20:
        raise RuntimeError("A potency state has too few cells for a stable density landscape")

    fig = plt.figure(figsize=(11.2, 6.2), facecolor="white")
    add_landscape(
        fig,
        [0.015, 0.515, 0.405, 0.430],
        coordinates,
        high_mask,
        1,
        "High potency state",
        HIGH_CMAP,
        0.395,
        0.755,
        0.585,
    )
    add_landscape(
        fig,
        [0.015, 0.045, 0.405, 0.430],
        coordinates,
        low_mask,
        n_states,
        "Low potency state",
        LOW_CMAP,
        0.395,
        0.285,
        0.115,
    )

    ax = fig.add_axes([0.445, 0.075, 0.535, 0.830], projection="3d")
    order = np.argsort(sr)
    points = ax.scatter(
        diffusion_display[order, 0],
        diffusion_display[order, 1],
        diffusion_display[order, 2],
        c=sr[order],
        cmap=SR_CMAP,
        s=7.5,
        alpha=0.86,
        linewidths=0,
        depthshade=False,
        rasterized=True,
    )
    ax.scatter(
        *diffusion_display[root],
        marker="^",
        s=78,
        facecolors="none",
        edgecolors="#D73027",
        linewidths=1.5,
        depthshade=False,
    )
    root_label = diffusion_display[root] + np.asarray([0.025, 0.015, 0.055])
    ax.text(
        *root_label,
        "   Root Cell",
        color="#D73027",
        fontsize=8.4,
        fontweight="bold",
        zorder=10,
    )
    terminal_labels = ["Predicted\nTerminal\nCell", "Predicted\nTerminal\nCell"]
    terminal_offsets = [
        np.asarray([-0.145, -0.135, 0.105]),
        np.asarray([0.075, 0.025, 0.145]),
    ]
    for terminal, label, offset in zip(terminals, terminal_labels, terminal_offsets):
        ax.scatter(
            *diffusion_display[terminal],
            marker="D",
            s=68,
            facecolors="none",
            edgecolors="#D73027",
            linewidths=1.4,
            depthshade=False,
        )
        label_position = diffusion_display[terminal] + offset
        ax.text(
            *label_position,
            label,
            color="#D73027",
            fontsize=7.5,
            fontweight="bold",
            zorder=10,
        )

    ax.set_title(
        "Diffusion Map Colored by SR Values",
        fontsize=13.6,
        fontweight="bold",
        pad=10,
    )
    ax.set_xlabel("DC1", fontsize=9.5, labelpad=1)
    ax.set_ylabel("DC2", fontsize=9.5, labelpad=1)
    ax.set_zlabel("DC3", fontsize=9.5, labelpad=1)
    ax.view_init(elev=24, azim=-51)
    ax.set_box_aspect((1.0, 1.0, 0.92))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("#D5D5D5")
        axis.pane.set_edgecolor("#B0B0B0")
        axis._axinfo["grid"]["color"] = (1, 1, 1, 1)
        axis._axinfo["grid"]["linewidth"] = 1.1
        axis.set_ticklabels([])
        axis.set_tick_params(length=0, pad=-2)
    ax.grid(True)

    color_cax = fig.add_axes([0.520, 0.285, 0.023, 0.265])
    colorbar = fig.colorbar(points, cax=color_cax)
    ticks = np.linspace(np.quantile(sr, 0.02), np.quantile(sr, 0.98), 5)
    colorbar.set_ticks(ticks)
    colorbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    colorbar.ax.tick_params(labelsize=7.5, width=0.7, length=3.0, pad=2)
    colorbar.outline.set_linewidth(0.8)
    fig.text(0.5315, 0.570, "SR", fontsize=11.5, fontweight="bold", ha="center")

    fig.text(0.009, 0.970, "G", fontsize=27, fontweight="normal", va="top", ha="left")
    fig.savefig(OUTPUT_STEM.with_suffix(".pdf"), dpi=600, facecolor="white")
    fig.savefig(OUTPUT_STEM.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)


def export_results(
    cells: pd.DataFrame,
    sr: np.ndarray,
    potency_states: np.ndarray,
    gmm_audit: pd.DataFrame,
    default_gmm_audit: pd.DataFrame,
    mapped_genes: pd.DataFrame,
    hvg_audit: pd.DataFrame,
    diffusion_rule: str,
    diffusion_raw: np.ndarray,
    diffusion_display: np.ndarray,
    diffusion_eigenvalues: np.ndarray,
    diffusion_signs: list[int],
    dpt: np.ndarray,
    root: int,
    terminals: np.ndarray,
    largest_eigenvalue: float,
    maximum_entropy_rate: float,
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output = cells.copy()
    output["signaling_entropy_rate"] = sr
    output["logit2_signaling_entropy_rate"] = np.log2(sr / (1 - sr))
    output["potency_state"] = [f"PS{state}" for state in potency_states]
    for column in range(3):
        output[f"DC{column + 1}_raw"] = diffusion_raw[:, column]
        output[f"DC{column + 1}_display"] = diffusion_display[:, column]
    output["diffusion_pseudotime_from_max_SR_root"] = dpt
    output["trajectory_role"] = "intermediate"
    output.loc[root, "trajectory_role"] = "root"
    for rank, terminal in enumerate(terminals, start=1):
        output.loc[terminal, "trajectory_role"] = f"predicted_terminal_{rank}"
    output.to_csv(
        RESULT_DIR / "Fig3G_epithelial_cells_SR_diffusion.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )

    state_counts = (
        output.groupby(["epithelial_cluster", "potency_state"], observed=False)
        .size()
        .rename("n_cells")
        .reset_index()
    )
    state_counts["fraction_within_cluster"] = state_counts["n_cells"] / state_counts.groupby(
        "epithelial_cluster", observed=False
    )["n_cells"].transform("sum")
    cluster_sr = (
        output.groupby("epithelial_cluster", observed=False)["signaling_entropy_rate"]
        .agg(["count", "mean", "median", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n_cells",
                "mean": "mean_SR",
                "median": "median_SR",
                "std": "sd_SR",
            }
        )
    )
    state_wide = state_counts.pivot(
        index="epithelial_cluster", columns="potency_state", values="fraction_within_cluster"
    ).fillna(0)
    state_wide.columns = [f"fraction_{column}" for column in state_wide.columns]
    cluster_summary = cluster_sr.merge(state_wide.reset_index(), on="epithelial_cluster")
    cluster_summary.to_csv(
        RESULT_DIR / "Fig3G_epithelial_SR_by_cluster.csv",
        index=False,
        encoding="utf-8-sig",
    )

    patient_summary = (
        output.groupby(
            ["patient_id", "sample_id", "disease_status", "epithelial_cluster"],
            observed=False,
        )["signaling_entropy_rate"]
        .agg(["count", "mean", "median"])
        .reset_index()
        .rename(
            columns={"count": "n_cells", "mean": "mean_SR", "median": "median_SR"}
        )
    )
    patient_summary.to_csv(
        RESULT_DIR / "Fig3G_epithelial_patient_cluster_SR.csv",
        index=False,
        encoding="utf-8-sig",
    )

    combined_gmm_audit = pd.concat([gmm_audit, default_gmm_audit], ignore_index=True)
    combined_gmm_audit.to_csv(
        RESULT_DIR / "Fig3G_epithelial_potency_GMM_BIC_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )
    mapped_genes.to_csv(
        RESULT_DIR / "Fig3G_epithelial_LandSCENT_PPI_gene_mapping.csv",
        index=False,
        encoding="utf-8-sig",
    )
    hvg_audit.to_csv(
        RESULT_DIR / "Fig3G_epithelial_diffusion_gene_audit.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )

    roles = [root, *terminals.tolist()]
    endpoint_table = output.iloc[roles][
        [
            "cell_id",
            "patient_id",
            "sample_id",
            "disease_status",
            "epithelial_cluster",
            "signaling_entropy_rate",
            "potency_state",
            "diffusion_pseudotime_from_max_SR_root",
            "trajectory_role",
            "DC1_raw",
            "DC2_raw",
            "DC3_raw",
        ]
    ]
    endpoint_table.to_csv(
        RESULT_DIR / "Fig3G_epithelial_root_and_terminals.csv",
        index=False,
        encoding="utf-8-sig",
    )

    clean_mask = ~output["epithelial_cluster"].isin(QC_CLUSTERS).to_numpy()
    clean_root = int(np.flatnonzero(clean_mask)[np.argmax(sr[clean_mask])])
    rho_genes = spearmanr(sr, output["n_genes_by_counts"], nan_policy="omit")
    rho_counts = spearmanr(sr, output["total_counts"], nan_policy="omit")
    cluster_groups = [
        output.loc[output["epithelial_cluster"].eq(cluster), "signaling_entropy_rate"].to_numpy()
        for cluster in CLUSTERS
    ]
    kw = kruskal(*cluster_groups)
    sensitivity = pd.DataFrame(
        [
            {
                "check": "primary_root_all_C0-C8",
                "value": output.loc[root, "cell_id"],
                "detail": f"cluster={output.loc[root, 'epithelial_cluster']}; SR={sr[root]:.8f}",
            },
            {
                "check": "root_excluding_C5_C8",
                "value": output.loc[clean_root, "cell_id"],
                "detail": f"cluster={output.loc[clean_root, 'epithelial_cluster']}; SR={sr[clean_root]:.8f}",
            },
            {
                "check": "root_stable_after_excluding_C5_C8",
                "value": str(root == clean_root),
                "detail": "C5 mixed and C8 immune-like are retained in the primary panel",
            },
            {
                "check": "SR_vs_detected_genes_spearman",
                "value": float(rho_genes.statistic),
                "detail": f"p={rho_genes.pvalue:.6g}",
            },
            {
                "check": "SR_vs_total_counts_spearman",
                "value": float(rho_counts.statistic),
                "detail": f"p={rho_counts.pvalue:.6g}",
            },
            {
                "check": "cell_level_cluster_Kruskal_Wallis",
                "value": float(kw.statistic),
                "detail": f"p={kw.pvalue:.6g}; descriptive only, cells are not independent patients",
            },
        ]
    )
    sensitivity.to_csv(
        RESULT_DIR / "Fig3G_epithelial_sensitivity_audit.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "source_object": str(H5AD),
        "figure3a_cell_table": str(FIG3A_CELLS),
        "n_cells": int(len(output)),
        "clusters": CLUSTERS,
        "user_decision": "C5 and C8 retained in the primary Figure 3G",
        "method": "LandSCENT-compatible signaling entropy rate and diffusion-map trajectory",
        "expression_transform": "log2(expm1(source_log1p_expression) + 1.1)",
        "ppi_source": str(LANDSCENT_RDATA),
        "ppi_repository": "https://github.com/ChenWeiyan/LandSCENT",
        "ppi_repository_commit": "ea35ec91690d399d2cb9d8147b7d0d77ce16f6b4",
        "ppi_network_name": "net17Jan16.m",
        "ppi_nodes_after_symbol_mapping_and_largest_component": int(len(mapped_genes)),
        "ppi_undirected_edges_after_mapping": int(mapped_genes.attrs.get("n_edges", 0)),
        "ppi_largest_eigenvalue": largest_eigenvalue,
        "maximum_entropy_rate_ln_lambda": maximum_entropy_rate,
        "sr_formula": "sum_i[x_i(Ax)_i*ln((Ax)_i/x_i)] / sum_i[x_i(Ax)_i] / ln(lambda_max(A))",
        "primary_potency_model": (
            "different-variance Gaussian mixtures on log2(SR/(1-SR)); BIC over K=1..2 "
            "to reproduce the reference Figure 3G high/low landscape"
        ),
        "default_maxPS5_sensitivity": (
            "BIC over K=1..5 is exported separately because LandSCENT default maxPS is 5"
        ),
        "primary_n_potency_states": int(potency_states.max()),
        "potency_state_counts": {
            f"PS{state}": int((potency_states == state).sum())
            for state in sorted(np.unique(potency_states))
        },
        "diffusion_gene_selection": diffusion_rule,
        "n_diffusion_genes": int(hvg_audit["selected_for_diffusion_map"].sum()),
        "diffusion_map": {
            "pca_components": N_PCA_COMPONENTS,
            "neighbors": K_NEIGHBORS,
            "kernel": "adaptive Gaussian kernel with alpha=1 density normalization",
            "eigenvalues": [float(value) for value in diffusion_eigenvalues],
            "display_orientation_signs": diffusion_signs,
        },
        "root_rule": "single cell with maximum SR among all retained C0-C8 cells",
        "root_cell": output.loc[root, "cell_id"],
        "root_cluster": output.loc[root, "epithelial_cluster"],
        "terminal_rule": (
            "two separated local maxima of graph-geodesic diffusion pseudotime from the SR root"
        ),
        "terminal_cells": output.iloc[terminals]["cell_id"].tolist(),
        "visual_reference": "s41746-025-02297-4 Figure 3G",
        "random_seed": SEED,
        "interpretation_boundary": (
            "SR is an expression/PPI-based differentiation-potency proxy, not a lineage-tracing "
            "measurement. The diffusion map is a static transcriptional-state geometry. Root and "
            "terminal labels are computational predictions. C5 is mixed and C8 is immune-like."
        ),
    }
    (RESULT_DIR / "Fig3G_epithelial_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    print("Loading official LandSCENT net17Jan16 PPI...", flush=True)
    official_adjacency, official_genes = load_official_ppi()
    print(
        f"  official nodes={official_adjacency.shape[0]:,}; "
        f"edges={official_adjacency.nnz // 2:,}",
        flush=True,
    )
    print("Loading fixed Figure 3A epithelial cells and expression...", flush=True)
    cells, delta_all, expression_genes = load_cells_and_expression()
    print(f"  cells={len(cells):,}; genes={len(expression_genes):,}", flush=True)

    print("Mapping expression genes to the official PPI largest component...", flush=True)
    adjacency, expression_columns, entrez_ids, mapped_genes = integrate_expression_and_ppi(
        official_adjacency, official_genes, expression_genes
    )
    del official_adjacency
    mapped_genes.attrs["n_edges"] = int(adjacency.nnz // 2)
    print(f"  mapped nodes={adjacency.shape[0]:,}; edges={adjacency.nnz // 2:,}", flush=True)

    print("Calculating normalized signaling entropy rates...", flush=True)
    sr, largest_eigenvalue, maximum_entropy_rate = calculate_signaling_entropy(
        delta_all[:, expression_columns].tocsr(), adjacency
    )
    print(
        f"  SR range={sr.min():.6f}-{sr.max():.6f}; median={np.median(sr):.6f}",
        flush=True,
    )

    print("Inferring binary reference-style potency states and maxPS=5 sensitivity...", flush=True)
    potency_states, gmm_audit, _ = fit_potency_states(
        sr, PRIMARY_MAX_POTENCY_STATES, "primary_reference_binary_maxPS2"
    )
    _, default_gmm_audit, _ = fit_potency_states(
        sr, DEFAULT_MAX_POTENCY_STATES, "LandSCENT_default_maxPS5_sensitivity"
    )
    print(
        "  primary "
        + ", ".join(
            f"PS{state}={(potency_states == state).sum():,}"
            for state in sorted(np.unique(potency_states))
        ),
        flush=True,
    )

    print("Selecting LandSCENT diffusion genes and constructing diffusion map...", flush=True)
    selected_columns, hvg_audit, diffusion_rule = select_diffusion_genes(
        delta_all, expression_genes
    )
    print(f"  selected genes={len(selected_columns):,}", flush=True)
    diffusion_raw, graph, diffusion_eigenvalues, _ = build_diffusion_map(
        delta_all, selected_columns
    )
    umap = cells[["umap_1", "umap_2"]].to_numpy(dtype=float)
    diffusion_display, diffusion_signs = orient_and_scale_diffusion(diffusion_raw, umap)
    clusters = cells["epithelial_cluster"].astype(str).to_numpy()
    root, terminals, dpt = select_root_and_terminals(
        sr, graph, diffusion_display, clusters
    )
    print(
        f"  root={cells.loc[root, 'cell_id']} ({clusters[root]}); "
        + ", ".join(
            f"terminal{rank}={cells.loc[cell, 'cell_id']} ({clusters[cell]})"
            for rank, cell in enumerate(terminals, start=1)
        ),
        flush=True,
    )

    export_results(
        cells,
        sr,
        potency_states,
        gmm_audit,
        default_gmm_audit,
        mapped_genes,
        hvg_audit,
        diffusion_rule,
        diffusion_raw,
        diffusion_display,
        diffusion_eigenvalues,
        diffusion_signs,
        dpt,
        root,
        terminals,
        largest_eigenvalue,
        maximum_entropy_rate,
    )
    plot_figure(cells, potency_states, diffusion_display, sr, root, terminals)
    print(f"Saved {OUTPUT_STEM.with_suffix('.pdf')}", flush=True)
    print(f"Saved {OUTPUT_STEM.with_suffix('.png')}", flush=True)


if __name__ == "__main__":
    main()
