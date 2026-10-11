"""Render verified hdWGCNA outputs in the reference Figure 4 composition."""
from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import hypergeom, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results/paper9/figure4_epithelial_hdwgcna"
FIG = ROOT / "figures/paper9/figure4_epithelial_hdwgcna"
COLORS = ["#377EB8", "#E64B35", "#7AC943", "#E5DD35", "#FA9A91", "#7752A1",
          "#171717", "#F4A3BB", "#AD60AD", "#4DBBB3", "#F5A130", "#80B1D3",
          "#8DD3C7", "#B15928", "#BC80BD", "#66A61E"]
SEED = 20260905
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": .7,
                     "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.facecolor": "white"})


def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1]
    out = np.empty(len(p)); out[order] = np.minimum(q, 1)
    return out


def mkey(m):
    return int(m[1:])


def subrect(rect, x, y, w, h):
    a, b, c, d = rect
    return [a+x*c, b+y*d, w*c, h*d]


def axes(fig, rect):
    return fig.add_axes(rect)


def letter(fig, rect, label):
    fig.text(rect[0], rect[1]+rect[3], label, fontsize=17, va="top", ha="left")


def clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def load_data():
    cells = pd.read_csv(RES / "input/cells.csv")
    me = pd.read_csv(RES / "cell_module_eigengenes.csv").set_index("r_cell_id")
    mods = pd.read_csv(RES / "gene_modules.csv")
    names = sorted(set(mods.module)-{"grey"}, key=mkey)
    scores = me.reindex(cells.r_cell_id)[names].reset_index(drop=True)
    scores = (scores-scores.mean()) / scores.std(ddof=0)
    genes = pd.read_csv(RES / "tom_gene_order.csv").gene.tolist()
    n = len(genes)
    tom = np.memmap(RES / "tom_float32.bin", dtype="<f4", mode="r", shape=(n, n), order="F")
    assert np.isfinite(tom).all() and np.allclose(tom, tom.T, atol=1e-6)
    assert np.isfinite(scores).all().all() and not scores.isna().any().any()
    idx = {g: i for i, g in enumerate(genes)}
    mods["gene_index"] = mods.gene_name.map(idx)
    mods["kME"] = [r[f"kME_{r['module']}"] for _, r in mods.iterrows()]
    mods["hub_rank"] = mods.groupby("module").kME.rank(ascending=False, method="first")
    colors = {m: COLORS[i % len(COLORS)] for i, m in enumerate(names)}
    mods["display_color"] = mods.module.map(colors).fillna("#BBBBBB")
    mods.to_csv(RES / "gene_modules_display.csv", index=False)
    um = pd.read_csv(RES / "module_umap.csv")
    um = um.drop(columns=[c for c in ("kME", "color", "hub", "module") if c in um])
    um = um.merge(mods, left_on="gene", right_on="gene_name", validate="one_to_one")
    return dict(cells=cells, scores=scores, mods=mods, names=names, colors=colors,
                tom=tom, idx=idx, um=um, power=pd.read_csv(RES / "soft_power.csv"))


def statistics(d):
    cells, scores, names = d["cells"], d["scores"], d["names"]
    joined = pd.concat([cells[["sample_id", "patient_id", "epithelial_cluster"]], scores], axis=1)
    sample = joined.groupby(["sample_id", "patient_id", "epithelial_cluster"]).agg(
        **{m: (m, "mean") for m in names}, n_cells=(names[0], "size")).reset_index()
    # Compare clusters within the same sample, then give each patient one paired contrast.
    valid = sample[sample.n_cells >= 10]
    rows = []
    for c in [f"C{i}" for i in range(9)]:
        target = valid[valid.epithelial_cluster.eq(c)].set_index("sample_id")
        other = valid[~valid.epithelial_cluster.isin([c, "C5", "C8"])].groupby("sample_id")[names].mean()
        common = target.index.intersection(other.index)
        diff = target.loc[common, names]-other.loc[common, names]
        diff["patient_id"] = target.loc[common, "patient_id"]
        patient = diff.groupby("patient_id")[names].mean()
        for m in names:
            vals = patient[m].dropna().to_numpy()
            p = wilcoxon(vals).pvalue if len(vals) >= 5 and np.any(vals != 0) else np.nan
            rows.append(dict(module=m, cluster=c, n_patients=len(vals),
                             paired_effect=np.mean(vals) if len(vals) else np.nan, pvalue=p))
    assoc = pd.DataFrame(rows); assoc["fdr"] = np.nan
    ok = assoc.pvalue.notna(); assoc.loc[ok, "fdr"] = bh(assoc.loc[ok, "pvalue"])
    assoc.to_csv(RES / "patient_paired_module_associations.csv", index=False)
    sample.to_csv(RES / "sample_cluster_module_eigengenes.csv", index=False)
    dot = []
    for m in names:
        mean = scores[m].groupby(cells.epithelial_cluster).mean()
        scaled = (mean-mean.mean()) / max(mean.std(ddof=0), 1e-8)
        for c in mean.index:
            vals = scores.loc[cells.epithelial_cluster.eq(c), m]
            dot.append(dict(module=m, cluster=c, mean_ME=mean[c], scaled_mean_ME=scaled[c],
                            percent_positive_ME=float((vals > 0).mean()*100)))
    dot = pd.DataFrame(dot)
    dot.to_csv(RES / "module_cluster_dotplot.csv", index=False)
    # C1 is the highest SR cluster in both cell-level and patient-balanced Figure 3H.
    peak = dot.loc[dot.groupby("module").mean_ME.idxmax()].set_index("module").cluster
    eligible_modules = peak[~peak.isin(["C5", "C8"])].index
    c1 = dot[dot.cluster.eq("C1") & dot.module.isin(eligible_modules)]
    selected = c1[c1.mean_ME.gt(0)].sort_values("mean_ME", ascending=False).head(2).module.tolist()
    sizes = d["mods"].groupby("module").size().sort_values(ascending=False)
    for m in sizes.index:
        if len(selected) < 4 and m in eligible_modules and m not in selected:
            selected.append(m)
    reasons = []
    for m in selected:
        reasons.append(dict(module=m, peak_cluster=peak[m],
                            role="C1-positive descriptive ME" if float(c1.set_index("module").loc[m,"mean_ME"])>0
                            else "complementary epithelial module by module size",
                            claim="Representative network, no claim of significant C1 enrichment"))
    pd.DataFrame(reasons).to_csv(RES/"representative_module_selection.csv",index=False)
    d.update(dot=dot, assoc=assoc, selected=selected)
    return d


def enrichment(d):
    pathways = []
    path = ROOT / "results/paper9/gene_sets/Enrichr.GO_Biological_Process_2023.gmt"
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            f = line.rstrip().split("\t")
            if len(f) >= 4:
                pathways.append((f[0], set(f[2:])))
    annotated = set().union(*(s for _, s in pathways))
    universe = set(d["mods"].gene_name) & annotated
    # Define testable terms before inspecting module overlaps, including zero-overlap tests in BH.
    terms = [(name, genes & universe) for name, genes in pathways if 5 <= len(genes & universe) <= 500]
    all_rows = []
    for m in d["names"]:
        query = set(d["mods"].loc[d["mods"].module.eq(m), "gene_name"]) & universe
        rows = []
        for name, genes in terms:
            overlap = query & genes
            p = float(hypergeom.sf(len(overlap)-1, len(universe), len(genes), len(query)))
            rows.append(dict(module=m, term=name, pvalue=p, overlap=len(overlap),
                             query_size=len(query), term_size=len(genes), universe_size=len(universe),
                             genes=";".join(sorted(overlap))))
        tab = pd.DataFrame(rows)
        tab["fdr"] = bh(tab.pvalue)
        all_rows.append(tab)
    go = pd.concat(all_rows, ignore_index=True)
    go["minus_log10_fdr"] = -np.log10(np.maximum(go.fdr, 1e-300))
    go.to_csv(RES / "GO_BP_2023_all_tests.csv.gz", index=False)
    display = go[go.fdr.lt(.05) & go.overlap.ge(3)].sort_values(["fdr", "pvalue", "term"])
    display = display.groupby("module", sort=False).head(3).copy()
    display["module_order"] = display.module.map({m: i for i, m in enumerate(d["names"])})
    display = display.sort_values(["module_order", "fdr", "pvalue"]).reset_index(drop=True)
    display.to_csv(RES / "GO_BP_2023_display.csv", index=False)
    d["go"] = display
    return d


def networks(d):
    um, tom = d["um"], d["tom"]
    indices = um.gene_index.to_numpy(int)
    sub = tom[np.ix_(indices, indices)].copy(); np.fill_diagonal(sub, 0)
    edges = set()
    for i in range(len(um)):
        for j in np.argpartition(sub[i], -3)[-3:]:
            if sub[i, j] > 0:
                edges.add(tuple(sorted((i, int(j)))))
    edges = np.array(sorted(edges), int)
    e = pd.DataFrame({"source": um.gene.to_numpy()[edges[:, 0]], "target": um.gene.to_numpy()[edges[:, 1]],
                      "weight_TOM": sub[edges[:, 0], edges[:, 1]]})
    e.to_csv(RES / "global_network_edges.csv", index=False)
    d["global_edges"] = edges
    d["global_weights"] = e.weight_TOM.to_numpy()
    # A compact companion network samples genes within modules; no intermodule links are imposed.
    rng = np.random.default_rng(SEED)
    inset = []
    for m in d["names"]:
        frame = d["mods"][d["mods"].module.eq(m)].sort_values("hub_rank")
        other = frame.iloc[5:]
        chosen = rng.choice(other.index.to_numpy(), min(13, len(other)), replace=False)
        inset.extend(frame.head(5).index); inset.extend(chosen)
    inset = d["mods"].loc[inset].reset_index(drop=True)
    ix = inset.gene_index.to_numpy(int)
    a = tom[np.ix_(ix, ix)].copy(); np.fill_diagonal(a, 0)
    graph = nx.Graph(); graph.add_nodes_from(range(len(inset)))
    for i in range(len(inset)):
        for j in np.argpartition(a[i], -6)[-6:]:
            if a[i, j] > 0:
                graph.add_edge(i, int(j), weight=float(a[i, j]))
    pos = nx.spring_layout(graph, seed=SEED, iterations=250, weight="weight")
    d.update(inset=inset, inset_graph=graph, inset_xy=np.array([pos[i] for i in range(len(inset))]))
    return d


def panel_a(fig, rect, d):
    letter(fig, rect, "A")
    p = d["power"]; beta = int(p.loc[p.selected, "Power"].iloc[0])
    keys = ["SFT.R.sq", "mean.k.", "median.k.", "max.k."]
    labels = ["Scale-free topology fit (R²)", "Mean Connectivity", "Median Connectivity", "Max Connectivity"]
    for i, (key, lab) in enumerate(zip(keys, labels)):
        x = .17 + (i % 2)*.45; y = .13 + (1-i//2)*.45
        ax = axes(fig, subrect(rect, x, y, .35, .34))
        v = p[key].to_numpy()
        if i == 0:
            v = -np.sign(p.slope.to_numpy())*v
            low = min(-.05, float(v.min())-.12)
            ax.set_ylim(low, 1.06)
            ax.axhspan(low, .8, color="#CDCDCD", zorder=0)
            ax.axhline(.8, color="black", ls=(0,(4,3)), lw=.8)
        else:
            ax.set_ylim(-float(v.max())*.08, float(v.max())*1.08)
        for power, value in zip(p.Power, v):
            ax.text(power, value, str(power), fontsize=4.6, ha="center", va="center")
        ax.axvline(beta, color="#222222", ls=(0,(4,3)), lw=.8)
        vbeta = v[p.Power.to_numpy() == beta][0]
        ax.scatter([beta], [vbeta], s=22, facecolors="none", edgecolors="black", lw=.8, zorder=3)
        ax.set_xlim(0, 32); ax.set_xticks([0,10,20,30])
        ax.set_xlabel("Soft Power Threshold", fontsize=5.7, labelpad=2)
        ax.set_ylabel(lab, fontsize=5.7, labelpad=2)
        ax.tick_params(labelsize=5, length=2, pad=1)


def panel_b(fig, rect, d):
    letter(fig, rect, "B")
    nr = math.ceil(len(d["names"])/3)
    xy = d["cells"][["umap_1", "umap_2"]].to_numpy()
    for i, m in enumerate(d["names"]):
        col, row = i % 3, i // 3
        ax = axes(fig, subrect(rect, .10+col*.30, .05+(nr-1-row)*.88/nr, .25, .78/nr))
        v = d["scores"][m].to_numpy()
        cmap = LinearSegmentedColormap.from_list(m, ["#D6D6D6", "#FAFAFA", d["colors"][m]])
        low,high = np.quantile(v,[.02,.98])
        order = np.argsort(v)
        ax.scatter(xy[order,0], xy[order,1], c=v[order], cmap=cmap, vmin=low, vmax=high,
                   s=.8, lw=0, rasterized=True)
        ax.set_aspect("equal"); clean(ax)
        ax.set_title(m, loc="center", fontsize=7, fontweight="bold", pad=1)
        cax = ax.inset_axes([1.01,.18,.055,.55])
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(low,high), cmap=cmap), cax=cax)
        cb.set_ticks([low,high]); cb.set_ticklabels(["-", "+"])
        cb.ax.tick_params(length=0, labelsize=6, pad=1); cb.outline.set_visible(False)


def network_lines(ax, xy, edges, weights, modules, colors, alpha=.15, width=.3, base_alpha=.025):
    w = np.asarray(weights)
    norm = (w-w.min()) / max(np.ptp(w), 1e-9)
    rgba = []
    for (i,j), strength in zip(edges, norm):
        color = colors[modules[i]] if modules[i] == modules[j] else "#999999"
        rgba.append(to_rgba(color, base_alpha+alpha*strength))
    ax.add_collection(LineCollection(xy[edges], colors=rgba, linewidths=.06+width*norm, zorder=0))


def repel_labels(ax, xy, selected, labels, fontsize=5.2):
    # Deterministic screen-space label relaxation; positions do not change gene coordinates.
    texts = [ax.text(xy[i,0], xy[i,1], labels[i], fontsize=fontsize, fontstyle="italic", zorder=4)
             for i in selected]
    ax.figure.canvas.draw()
    renderer = ax.figure.canvas.get_renderer()
    for iteration in range(60):
        boxes = [t.get_window_extent(renderer).expanded(1.08,1.12) for t in texts]
        moved = False
        for i in range(len(texts)):
            for j in range(i):
                if boxes[i].overlaps(boxes[j]):
                    pi = ax.transData.transform(texts[i].get_position())
                    pj = ax.transData.transform(texts[j].get_position())
                    sign = 1 if pi[1] >= pj[1] else -1
                    texts[i].set_position(ax.transData.inverted().transform(pi+[0, sign*1.5]))
                    texts[j].set_position(ax.transData.inverted().transform(pj-[0, sign*1.5]))
                    moved = True
        if not moved:
            break
    return texts


def panel_c(fig, rect, d, inset_rect=None):
    letter(fig, rect, "C")
    ax = axes(fig, subrect(rect, .035,.035,.94,.90))
    um = d["um"]
    xy = um[["UMAP1", "UMAP2"]].to_numpy()
    e = d["global_edges"]
    network_lines(ax, xy, e, d["global_weights"], um.module.to_numpy(), d["colors"], alpha=.19, width=.28)
    size = 1.1 + 8*np.maximum(um.kME.to_numpy(), 0)**3
    ax.scatter(xy[:,0],xy[:,1], s=size, c=um.display_color, lw=0, zorder=2, rasterized=True)
    ax.margins(.12); clean(ax)
    selected = um.index[um.hub_rank.le(2)].tolist()
    repel_labels(ax, xy, selected, um.gene.tolist(), fontsize=5.5)
    if inset_rect is not None:
        ax2 = axes(fig, inset_rect)
        frame, graph, xy2 = d["inset"], d["inset_graph"], d["inset_xy"]
        edges = np.array(list(graph.edges()), int)
        weights = np.array([graph[i][j]["weight"] for i,j in edges])
        network_lines(ax2, xy2, edges, weights, frame.module.to_numpy(), d["colors"], alpha=.14, width=.35,base_alpha=.10)
        ax2.scatter(xy2[:,0], xy2[:,1], s=np.where(frame.hub_rank.le(5), 18, 4),
                    c=frame.display_color, lw=0, zorder=2)
        ax2.margins(.12); clean(ax2)
        repel_labels(ax2, xy2, frame.index[frame.hub_rank.le(3)].tolist(), frame.gene_name.tolist(), fontsize=4.8)


def panel_d(fig, rect, d):
    letter(fig, rect, "D")
    ax = axes(fig, subrect(rect, .18,.20,.60,.69))
    tab = d["dot"]
    xx = tab.cluster.str[1:].astype(int)
    yy = tab.module.map({m:i for i,m in enumerate(d["names"])})
    cmap = LinearSegmentedColormap.from_list("ME", ["#88BBDD", "#F7F7F7", "#D72F28"])
    sc = ax.scatter(xx, yy, s=5+tab.percent_positive_ME*.8, c=tab.scaled_mean_ME,
                    cmap=cmap, vmin=-1, vmax=2, linewidths=0)
    ax.set_xticks(range(9)); ax.set_xticklabels([f"C{i}" for i in range(9)], rotation=90, fontsize=5.2)
    ax.set_yticks(range(len(d["names"]))); ax.set_yticklabels(d["names"], fontsize=5.2)
    ax.set_xlim(-.55,8.55); ax.set_ylim(-.55,len(d["names"])-.45)
    ax.set_xticks(np.arange(-.5,9), minor=True); ax.set_yticks(np.arange(-.5,len(d["names"])),minor=True)
    ax.grid(which="minor",color="#DDDDDD",lw=.5); ax.set_axisbelow(True)
    ax.tick_params(which="both",length=0,pad=1)
    cax = axes(fig, subrect(rect, .84,.55,.025,.23))
    cb = fig.colorbar(sc,cax=cax); cb.set_ticks([-1,0,1,2]); cb.ax.tick_params(labelsize=4.5,length=0,pad=1)
    cb.outline.set_visible(False)
    fig.text(*subrect(rect,.81,.80,0,0)[:2], "Average ME", fontsize=5)
    lax = axes(fig,subrect(rect,.82,.13,.18,.29)); clean(lax); lax.set_xlim(0,1); lax.set_ylim(0,1)
    lax.text(0,1,"Positive ME (%)",fontsize=4.8)
    for i,p in enumerate([25,50,75]):
        lax.scatter([.15],[.75-i*.25],s=5+p*.8,c="black",lw=0)
        lax.text(.40,.75-i*.25,str(p),fontsize=4.8,va="center")


def panel_e(fig, rect, d):
    letter(fig, rect, "E")
    edge_rows = []
    for k,m in enumerate(d["selected"]):
        ax = axes(fig,subrect(rect,.04+(k%2)*.49,.04+(1-k//2)*.47,.44,.42))
        frame = d["mods"][d["mods"].module.eq(m)].sort_values("hub_rank").head(25).reset_index(drop=True)
        n = len(frame); n_inner = min(5,n)
        inner = np.linspace(np.pi/2,np.pi/2+2*np.pi,n_inner,endpoint=False)
        outer = np.linspace(np.pi/2,np.pi/2+2*np.pi,n-n_inner,endpoint=False)
        xy = np.vstack([.48*np.column_stack([np.cos(inner),np.sin(inner)]),
                        np.column_stack([np.cos(outer),np.sin(outer)])])
        ii = frame.gene_index.to_numpy(int)
        mat = d["tom"][np.ix_(ii,ii)]
        pairs = np.column_stack(np.triu_indices(n,1)); weights = mat[pairs[:,0],pairs[:,1]]
        network_lines(ax,xy,pairs,weights,np.repeat(m,n),d["colors"],alpha=.48,width=.6)
        ax.scatter(xy[:,0],xy[:,1],s=13,c=d["colors"][m],edgecolors="#444444",linewidths=.25,zorder=2)
        for i,g in enumerate(frame.gene_name):
            tx,ty = xy[i]*(1.13 if i<n_inner else 1.11)
            ax.text(tx,ty,g,fontsize=5.1,fontstyle="italic",ha="center",va="center")
        ax.set_xlim(-1.35,1.35); ax.set_ylim(-1.3,1.3); ax.set_aspect("equal"); clean(ax)
        ax.set_title(m,fontsize=7,fontweight="bold",pad=1)
        for (i,j),w in zip(pairs,weights):
            edge_rows.append(dict(module=m,source=frame.gene_name[i],target=frame.gene_name[j],weight_TOM=float(w)))
    pd.DataFrame(edge_rows).to_csv(RES / "selected_hub_network_edges.csv",index=False)


def panel_f(fig,rect,d):
    letter(fig,rect,"F")
    go = d["go"]
    ax = axes(fig,subrect(rect,.58,.10,.31,.82))
    yy = np.arange(len(go))[::-1]
    xx = go.module.map({m:i for i,m in enumerate(d["names"])})
    value = np.minimum(go.minus_log10_fdr.to_numpy(),12)
    ax.scatter(xx,yy,s=3+value*2.5,c=go.module.map(d["colors"]),lw=0)
    labels = [textwrap.fill(re.sub(r"\s*\(GO:\d+\)","",t),width=38) for t in go.term]
    ax.set_yticks(yy); ax.set_yticklabels(labels,fontsize=5.8,linespacing=1.02)
    ax.set_xticks(range(len(d["names"]))); ax.set_xticklabels(d["names"],rotation=50,ha="right",fontsize=5)
    ax.set_xlim(-.6,len(d["names"])-.4); ax.set_ylim(-.6,len(go)+2)
    ax.tick_params(length=2,width=.4,pad=1)
    ax.set_title("GO_Biological_Process_2023",fontsize=6.8,fontweight="bold",pad=5)
    for p in [2,5,8,12]:
        ax.scatter([],[],s=3+p*2.5,c="#111111",label=str(p))
    ax.legend(title="Enrichment\n(-log10 FDR)",fontsize=4.5,title_fontsize=4.6,frameon=False,
              loc="upper left",borderpad=.2,handletextpad=.2,labelspacing=.25)


def save(fig,stem):
    fig.savefig(FIG/f"{stem}.pdf",dpi=450)
    fig.savefig(FIG/f"{stem}.png",dpi=450)
    plt.close(fig)


def main():
    FIG.mkdir(parents=True,exist_ok=True)
    d = networks(enrichment(statistics(load_data())))
    print("Modules:",d["mods"].groupby("module").size().to_dict(),flush=True)
    print("Representative epithelial modules:",d["selected"],flush=True)
    print(d["assoc"][d["assoc"].cluster.eq("C1")].to_string(index=False),flush=True)
    specs = [("A",panel_a,(4.6,4.4)),("B",panel_b,(4.3,4.5)),("C",panel_c,(7.0,7.5)),
             ("D",panel_d,(4.8,3.3)),("E",panel_e,(7.5,7.0)),("F",panel_f,(6.2,8.0))]
    for name,func,size in specs:
        fig = plt.figure(figsize=size)
        if name == "C":
            func(fig,[.025,.28,.95,.70],d,inset_rect=[.05,.035,.90,.23])
        else:
            func(fig,[.025,.025,.95,.95],d)
        save(fig,f"Fig4{name}_hdWGCNA_reference_style")
    fig = plt.figure(figsize=(12,13.2))
    panel_a(fig,[.025,.66,.31,.32],d)
    panel_b(fig,[.335,.66,.30,.32],d)
    panel_c(fig,[.645,.63,.33,.35],d,inset_rect=[.055,.48,.57,.16])
    panel_d(fig,[.65,.46,.32,.17],d)
    panel_e(fig,[.03,.035,.60,.42],d)
    panel_f(fig,[.60,.025,.39,.425],d)
    save(fig,"Figure4_hdwgcna_reference_layout")
    summary = dict(selected_modules=d["selected"],reference_cluster="C1",n_display_network_genes=len(d["um"]),
                   n_display_GO_terms=len(d["go"]),GO_universe=int(d["go"].universe_size.iloc[0]),
                   GO_method="Local GO BP 2023 ORA, network-gene background including grey; BH over every pre-defined testable term; FDR<0.05 only",
                   plot_D="mean single-cell ME standardized across clusters; area is fraction of cells with z-ME>0; accurate ME labels",
                   plot_F="Same module-colored dotplot structure; size encodes -log10 FDR, not Enrichr combined score",
                   representative_module_rule="Two C1-positive mean-ME modules, then largest complementary modules whose peak is not C5/C8; representative networks, not four claimed C1-enriched modules")
    (RES/"plot_metadata.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print("Saved",FIG,flush=True)


if __name__ == "__main__":
    main()
