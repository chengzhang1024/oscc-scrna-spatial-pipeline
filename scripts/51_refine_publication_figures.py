"""Refine five figures from frozen results; never refit models or rewrite source data."""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import re
import textwrap
from pathlib import Path
import fitz
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"figures/paper9"
OUT=BASE/"visual_revision_20260906"
PAN=OUT/"panels"
SUP=OUT/"supplementary"
R3=ROOT/"results/paper9/figure3_epithelial"
R4=ROOT/"results/paper9/figure4_epithelial_hdwgcna"
R5=ROOT/"results/paper9/figure5_survival"
BLUE="#397FA6"; RED="#C75162"; GREEN="#438873"; INK="#252525"
MODCOL=["#367BA8","#CF594E","#6DAD58","#B3A630","#DF8B93","#8260A5","#424242","#CB76A5"]
USED={}; MANIFEST=[]
plt.rcParams.update({"font.family":"Arial","font.size":9,"axes.titlesize":10,
                     "axes.labelsize":8.5,"xtick.labelsize":8,"ytick.labelsize":8,
                     "axes.linewidth":.65,"pdf.fonttype":42,"ps.fonttype":42,
                     "svg.fonttype":"none","text.color":INK,"axes.labelcolor":INK})


def track(path):
    path=Path(path)
    if str(path) not in USED:
        USED[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
    return path


def read(path,**kwargs):
    return pd.read_csv(track(path),**kwargs)


def jsonread(path):
    return json.loads(track(path).read_text(encoding="utf-8"))


def style(ax,grid=False):
    ax.spines[["top","right"]].set_visible(False)
    ax.tick_params(length=3,width=.6,pad=3)
    if grid:
        ax.grid(axis="x",color="#E7E7E7",lw=.55)
        ax.set_axisbelow(True)


def export(fig,name,directory=PAN):
    path=directory/f"{name}.pdf"
    fig.savefig(path,dpi=450,facecolor="white")
    fig.savefig(path.with_suffix(".png"),dpi=220,facecolor="white")
    plt.close(fig)
    return path


def wrapped(label,width=32):
    return textwrap.fill(label,width=width,break_long_words=False)


def clean_go(term):
    return re.sub(r"\s*\(GO:\d+\)\s*$","",term).strip()


def separate_network_labels(ax,texts):
    ax.figure.canvas.draw()
    renderer=ax.figure.canvas.get_renderer()
    original=[text.get_position() for text in texts]
    # Relax only text bounding boxes in screen coordinates; nodes and edges remain fixed.
    for _ in range(120):
        moved=False
        for i in range(len(texts)):
            for j in range(i):
                a=texts[i].get_window_extent(renderer).expanded(1.08,1.16)
                b=texts[j].get_window_extent(renderer).expanded(1.08,1.16)
                if not a.overlaps(b):continue
                pi=ax.transData.transform(texts[i].get_position())
                pj=ax.transData.transform(texts[j].get_position())
                overlap_x=min(a.x1,b.x1)-max(a.x0,b.x0)
                overlap_y=min(a.y1,b.y1)-max(a.y0,b.y0)
                dim=0 if overlap_x<overlap_y else 1
                step=np.zeros(2);step[dim]=(1 if pi[dim]>=pj[dim] else -1)*(min(overlap_x,overlap_y)/2+.3)
                texts[i].set_position(ax.transData.inverted().transform(pi+step))
                texts[j].set_position(ax.transData.inverted().transform(pj-step));moved=True
        if not moved:break
    for text,anchor in zip(texts,original):
        if np.linalg.norm(ax.transData.transform(text.get_position())-ax.transData.transform(anchor))>3:
            ax.annotate("",xy=anchor,xytext=text.get_position(),arrowprops=dict(arrowstyle="-",color="#A4A4A4",lw=.35))


def import_module(name,filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/filename)
    module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def prepare_pdf(path,labels=(),clip=None):
    src=fitz.open(track(path)); page=src[0]
    type3={f[3].split("+")[-1] for f in page.get_fonts() if f[2]=="Type3"}
    replacements=[]
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines",[]):
            for span in line["spans"]:
                if span["text"].strip() in labels and span["size"]>=9:
                    page.add_redact_annot(fitz.Rect(span["bbox"]),fill=False)
                elif span["font"] in type3:
                    page.add_redact_annot(fitz.Rect(span["bbox"]),fill=False)
                    replacements.append(span)
    page.apply_redactions(images=0,graphics=0)
    for span in replacements:
        rgb=tuple(((span["color"]>>shift)&255)/255 for shift in [16,8,0])
        page.insert_text(span["origin"],span["text"],fontsize=span["size"],fontname="RevisionArial",
                         fontfile=matplotlib.font_manager.findfont("Arial"),color=rgb)
    page.clean_contents()
    region=fitz.Rect(clip) if clip else page.rect
    pix=page.get_pixmap(matrix=fitz.Matrix(2,2),clip=region,alpha=False)
    img=np.frombuffer(pix.samples,np.uint8).reshape(pix.height,pix.width,pix.n)
    yy,xx=np.where(img[:,:,:3].min(axis=2)<248)
    content=fitz.Rect(region.x0+xx.min()/2,region.y0+yy.min()/2,
                      region.x0+(xx.max()+1)/2,region.y0+(yy.max()+1)/2)
    return src,(content+(-2,-2,2,2))&region


def put(doc,path,box,label=None,clip=None,remove=None,align="center"):
    src,crop=prepare_pdf(path,tuple(remove or ([label] if label else [])),clip)
    r=fitz.Rect(box)
    target=r+(13,15,-3,-3)
    factor=min(target.width/crop.width,target.height/crop.height)
    w,h=crop.width*factor,crop.height*factor
    left=target.x0+(target.width-w)/2
    top=target.y0 if align=="top" else target.y0+(target.height-h)/2
    exact=fitz.Rect(left,top,left+w,top+h)
    doc[0].show_pdf_page(exact,src,0,clip=crop)
    if label:
        doc[0].insert_text((r.x0,r.y0+14),label,fontsize=15,fontname="hebo",color=(.12,.12,.12))
    MANIFEST.append(dict(source=str(Path(path).relative_to(ROOT)),label=label,
                          crop=list(crop),destination=list(exact),scale=factor))
    src.close()


def save_doc(doc,name):
    doc.save(OUT/f"{name}.pdf",deflate=True,garbage=4)
    doc[0].get_pixmap(dpi=300).save(OUT/f"{name}.png")
    doc.close()


def fig1_go():
    data=read(ROOT/"results/paper9/Fig1H_selected_GO_pathways.csv")
    fig,axs=plt.subplots(1,2,figsize=(8.2,3.7))
    fig.subplots_adjust(left=.27,right=.98,bottom=.16,top=.86,wspace=1.85)
    for ax,(comp,color) in zip(axs,[("OLK vs Normal",GREEN),("OSCC vs Normal",RED)]):
        tab=data[data.comparison.eq(comp)].sort_values("display_order")
        ax.barh(np.arange(len(tab)),tab.enrichment_score,color=color,height=.52)
        ax.set_yticks(np.arange(len(tab))); ax.set_yticklabels([wrapped(t,23) for t in tab.display_label],fontsize=8)
        ax.invert_yaxis(); ax.set_xlim(0,10); ax.set_xticks([0,5,10]); ax.axvline(-np.log10(.05),lw=.6,ls="--",color="#909090")
        ax.set_title(comp,fontsize=10,fontweight="bold",pad=9)
        ax.set_xlabel("-log10(FDR)"); style(ax,True)
    data.to_csv(PAN/"Fig1H_display_values.csv",index=False)
    return export(fig,"Fig1H_GO_comparison")


def figure1():
    h=fig1_go()
    marker=import_module("oldmarkers","17_plot_fig1b_marker_tme_maps.py")
    track(marker.MARKER_STATS)
    fig,ax=plt.subplots(figsize=(8.0,2.8));fig.subplots_adjust(left=.12,right=.98,bottom=.22,top=.73)
    mapper=marker.make_dotplot(ax)
    cb=fig.colorbar(mapper,cax=fig.add_axes([.68,.85,.26,.023]),orientation="horizontal",ticks=[-1,0,1,2.5])
    cb.ax.tick_params(labelsize=8,length=0,pad=2);cb.outline.set_visible(False)
    fig.text(.81,.925,"Scaled mean expression",ha="center",fontsize=9)
    ax.tick_params(axis="x",labelsize=9);ax.tick_params(axis="y",labelsize=10)
    b=export(fig,"Fig1B_marker_matrix")
    comp=read(ROOT/"results/paper9/composition_literature_v3_GSE181919_per_sample.csv")
    groups=["Normal","OLK","OSCC"];comp=comp[comp.disease_status.isin(groups)]
    means=comp.groupby("disease_status")[CELL_TYPES].mean().reindex(groups)
    roe=means.div(comp[CELL_TYPES].mean(),axis=1)
    roe.to_csv(PAN/"Fig1E_exact_roe.csv")
    fig,ax=plt.subplots(figsize=(4.3,2.6));fig.subplots_adjust(left=.14,right=.86,bottom=.35,top=.80)
    mat=np.log2(roe.to_numpy());im=ax.imshow(mat,cmap="RdBu_r",vmin=-3,vmax=3,aspect="auto")
    for y in range(3):
        for x in range(9):ax.text(x,y,f"{roe.iloc[y,x]:.2f}",fontsize=8.5,ha="center",va="center",color="white" if abs(mat[y,x])>1.5 else INK)
    ax.set_xticks(range(9));ax.set_xticklabels(SHORT,rotation=65,ha="right",fontsize=8.5)
    ax.set_yticks(range(3));ax.set_yticklabels(groups,fontsize=9.5);ax.tick_params(length=0,pad=3)
    ax.set_title("Sample-balanced Ro/e",fontsize=11,fontweight="bold",pad=8)
    cb=fig.colorbar(im,cax=fig.add_axes([.90,.36,.021,.40]),ticks=[-3,0,3]);cb.ax.tick_params(labelsize=8,length=0);cb.outline.set_visible(False)
    fig.text(.91,.84,"log2 Ro/e",fontsize=8,ha="center")
    e=export(fig,"Fig1E_composition_enrichment")
    med=comp.groupby("disease_status")[CELL_TYPES].median().reindex(groups)
    med.to_csv(PAN/"Fig1F_exact_median_fractions.csv")
    fig,ax=plt.subplots(figsize=(4.4,4.5));fig.subplots_adjust(left=.29,right=.93,bottom=.15,top=.83)
    for k,(group,color) in enumerate(zip(groups,["#5F8AA8","#E87A55","#5CA48E"])):
        y=np.arange(9)+(k-1)*.22
        ax.scatter(med.loc[group].to_numpy()*100,y,c=color,s=32,label=group,zorder=3)
    ax.set_yticks(range(9));ax.set_yticklabels(SHORT,fontsize=10);ax.invert_yaxis()
    ax.set_xlim(-1,65);ax.set_xticks([0,20,40,60]);ax.set_xlabel("Median cell fraction (%)",fontsize=10)
    ax.set_title("Patient-level composition",fontsize=11,fontweight="bold",pad=28)
    ax.legend(ncol=3,frameon=False,loc="lower center",bbox_to_anchor=(.5,1.01),fontsize=8,handletextpad=.25,columnspacing=.8)
    style(ax,True)
    f=export(fig,"Fig1F_patient_composition")
    doc=fitz.open();doc.new_page(width=864,height=1030)
    put(doc,BASE/"Normal_OLK_OSCC_celltypes_reference_style.pdf",(16,12,848,192),"A")
    put(doc,b,(16,201,628,452),"B")
    put(doc,BASE/"Fig1B_marker_TME_feature_maps.pdf",(633,205,848,452),clip=(840,0,1167.5,406.3))
    put(doc,BASE/"Fig1C_lineage_marker_heatmap.pdf",(16,462,293,786),"C")
    dynamics=BASE/"Fig1DEF_TME_dynamics.pdf"
    put(doc,dynamics,(298,462,589,653),"D",clip=(0,0,410,285),remove=["D"])
    put(doc,e,(298,656,589,786),"E")
    put(doc,f,(591,462,848,786),"F")
    put(doc,BASE/"Fig1G_epithelial_stage_transitions.pdf",(16,797,290,1017),"G")
    put(doc,h,(298,797,848,1017),"H")
    save_doc(doc,"Figure1_visual_revision")


CELL_TYPES=["B/Plasma cells","Dendritic cells","Endothelial cells","Epithelial cells","Fibroblasts",
            "Macrophages","Mast cells","Myocytes","NK/T cells"]
SHORT=["B/Plasma","Dendritic","Endothelial","Epithelial","Fibroblast","Macrophage","Mast","Myocyte","NK/T"]


def misty(early=True):
    file=ROOT/"results/spatial_misty"/("figure2c_early/early_misty_group_importance.csv" if early else "figure2i_advanced/advanced_misty_group_importance.csv")
    data=read(file)
    cmap=LinearSegmentedColormap.from_list("misty",["#ECF3F0","#68B79B","#245F51"]);cmap.set_bad("#FAFAFA")
    fig,axs=plt.subplots(1,3,figsize=(8.4,3.0))
    fig.subplots_adjust(left=.105,right=.935,bottom=.25,top=.83,wspace=.45)
    for ax,view,title in zip(axs,["intra","juxta","para"],["Intra","Juxta (5)","Para (15)"]):
        tab=data[data.view.eq(view)].copy()
        tab["shown"]=tab.median_importance_z.where(tab.stability_fraction.ge(.75)&tab.median_importance_z.ge(1))
        matrix=tab.pivot(index="target",columns="predictor",values="shown").reindex(index=CELL_TYPES[::-1],columns=CELL_TYPES).to_numpy()
        im=ax.imshow(np.ma.masked_invalid(matrix),cmap=cmap,vmin=1,vmax=2.5)
        ax.set_xticks(range(9));ax.set_xticklabels(SHORT,rotation=55,ha="right",fontsize=7)
        ax.set_yticks(range(9));ax.set_yticklabels(SHORT[::-1],fontsize=7)
        ax.set_xticks(np.arange(-.5,9),minor=True);ax.set_yticks(np.arange(-.5,9),minor=True)
        ax.grid(which="minor",color="#D7DDDB",lw=.45);ax.tick_params(which="both",length=0,pad=2)
        ax.set_title(title,fontsize=10,fontweight="bold",pad=7)
        for spine in ax.spines.values():spine.set_visible(False)
    cb=fig.colorbar(im,cax=fig.add_axes([.958,.37,.012,.34]),ticks=[1,1.5,2,2.5]);cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7,length=0)
    fig.text(.965,.755,"Importance",ha="center",fontsize=7)
    return export(fig,"Fig2C_MISTy" if early else "Fig2I_MISTy")


def figure2():
    c,i=misty(True),misty(False)
    f2=BASE/"figure2"
    doc=fitz.open();doc.new_page(width=864,height=880)
    for off,letters,suffix,heat in [(0,"ABCDEF","early_OSCC",c),(436,"GHIJKL","advanced_OSCC",i)]:
        a,b,cc,d,e,f=letters
        put(doc,f2/f"Fig2{a}_cell2location_{suffix}.pdf",(16,12+off,182,201+off),a,align="top")
        put(doc,f2/f"Fig2{b}_markers_{suffix}.pdf",(190,12+off,372,201+off),b,align="top")
        put(doc,heat,(379,12+off,848,201+off),cc)
        put(doc,f2/f"Fig2{d}_cell2location_abundance_{suffix}.pdf",(16,213+off,227,425+off),d)
        put(doc,f2/f"Fig2{e}_MISTy_intra_network_{suffix}.pdf",(235,213+off,393,425+off),e)
        put(doc,f2/f"Fig2{f}_spatial_niches_{suffix}.pdf",(401,213+off,848,425+off),f)
    save_doc(doc,"Figure2_visual_revision")


def fig3_details():
    out={}
    tab=read(R3/"Fig3F_epithelial_pathway_activity.csv",index_col=0)
    fig,ax=plt.subplots(figsize=(4.5,3.2));fig.subplots_adjust(left=.10,right=.88,bottom=.28,top=.91)
    im=ax.imshow(tab,cmap="RdBu_r",vmin=-2,vmax=2,aspect="auto")
    ax.set_yticks(range(9));ax.set_yticklabels(tab.index,fontsize=9)
    ax.set_xticks(range(6));ax.set_xticklabels(["Keratinization","EMT","IFN /\ninflammation","Hypoxia /\nglycolysis","DNA\nrepair","Cell\ncycle"],fontsize=7,rotation=30,ha="right")
    ax.tick_params(length=0,pad=4)
    for y in range(9):
        for x in range(6):
            v=tab.iloc[y,x];ax.text(x,y,f"{v:.2f}",ha="center",va="center",fontsize=7,color="white" if abs(v)>1.25 else INK)
    ax.set_title("Pathway activity",fontweight="bold",pad=8)
    cb=fig.colorbar(im,cax=fig.add_axes([.92,.31,.018,.52]),ticks=[-2,0,2]);cb.outline.set_visible(False)
    cb.ax.tick_params(length=0,labelsize=8)
    out["F"]=export(fig,"Fig3F_activity")
    h=read(R3/"Fig3H_epithelial_mean_SR_and_sensitivity.csv").sort_values("epithelial_cluster")
    fig,ax=plt.subplots(figsize=(4.5,2.5));fig.subplots_adjust(left=.16,right=.97,bottom=.24,top=.82)
    ax.scatter(np.arange(9),h.mean_SR,c=[RED if c=="C1" else BLUE for c in h.epithelial_cluster],s=32,zorder=3)
    ax.set_xticks(range(9));ax.set_xticklabels(h.epithelial_cluster,fontsize=8)
    ax.set_ylim(.84,.93);ax.set_yticks([.85,.88,.91]);ax.set_ylabel("Mean SR")
    ax.set_title("Mean signaling entropy",fontweight="bold",pad=9);style(ax)
    ax.grid(axis="y",color="#E5E5E5",lw=.6);ax.set_axisbelow(True)
    out["H"]=export(fig,"Fig3H_mean_SR")
    summary=read(R3/"Fig3I_epithelial_SR_distribution_summary.csv").sort_values("epithelial_cluster")
    boxes=[dict(label=r.epithelial_cluster,med=r.median_SR,q1=r.q1_SR,q3=r.q3_SR,
                 whislo=r.lower_whisker_SR,whishi=r.upper_whisker_SR,fliers=[]) for r in summary.itertuples()]
    fig,ax=plt.subplots(figsize=(4.6,2.6));fig.subplots_adjust(left=.15,right=.97,bottom=.23,top=.82)
    artists=ax.bxp(boxes,showfliers=False,patch_artist=True,widths=.55)
    for patch,c in zip(artists["boxes"],summary.epithelial_cluster):
        patch.set_facecolor(RED if c=="C1" else "#A8CBDD");patch.set_linewidth(.7)
    for line in artists["medians"]:line.set_color("#202020");line.set_linewidth(1)
    ax.set_ylim(.82,.955);ax.set_yticks([.84,.89,.94]);ax.set_ylabel("Single-cell SR")
    ax.set_title("SR distribution",fontweight="bold",pad=8);style(ax)
    out["I"]=export(fig,"Fig3I_SR_distribution")
    go=read(R3/"Fig3J_epithelial_high_SR_cluster_GO_enrichment_all.csv").sort_values("rank").head(10)
    fig,ax=plt.subplots(figsize=(6.2,4.25));fig.subplots_adjust(left=.65,right=.98,bottom=.13,top=.89)
    ax.barh(range(len(go)),go.minus_log10_fdr,color="#8065A3",height=.56)
    short_go=["Epidermis development","Cell migration regulation","Endogenous antigen / MHC-I / ER",
               "Endogenous antigen / MHC-I / ER / TAP-independent","Endogenous antigen / MHC-Ib",
               "Peptide antigen / MHC-Ib","Intermediate filament organization",
               "Negative regulation of NK cytotoxicity","Endogenous peptide antigen presentation",
               "Cell-substrate junction assembly"]
    go["short_display_label"]=short_go
    go.to_csv(PAN/"Fig3J_full_terms_and_short_labels.csv",index=False)
    ax.set_yticks(range(len(go)));ax.set_yticklabels([wrapped(t,39) for t in short_go],fontsize=9)
    ax.invert_yaxis();ax.set_xlim(0,5);ax.set_xticks([0,2,4]);style(ax,True)
    ax.set_xlabel("-log10(FDR)");ax.set_title("C1 marker GO",fontweight="bold",pad=10)
    out["J"]=export(fig,"Fig3J_C1_GO")
    return out


def figure3():
    extra=fig3_details();f3=BASE/"figure3_epithelial"
    paths={p.name[4]:p for p in f3.glob("Fig3*.pdf")}
    doc=fitz.open();doc.new_page(width=864,height=1065)
    put(doc,paths["A"],(16,12,463,181),"A")
    put(doc,paths["C"],(16,186,463,365),"C")
    put(doc,paths["B"],(475,12,848,365),"B")
    put(doc,paths["D"],(16,372,848,584),"D")
    put(doc,paths["E"],(16,590,607,768),"E")
    put(doc,extra["F"],(613,590,848,768),"F")
    put(doc,paths["G"],(16,777,230,1052),"G",clip=(0,0,360,446.4))
    put(doc,paths["G"],(235,777,506,934),clip=(365,0,806.4,446.4),remove=["G"])
    put(doc,extra["H"],(235,943,506,1052),"H")
    put(doc,extra["I"],(512,777,848,887),"I")
    put(doc,extra["J"],(512,892,848,1052),"J")
    save_doc(doc,"Figure3_visual_revision")


def figure4():
    module=import_module("fig4_original","scripts/42_plot_figure4_hdwgcna.py")
    mods=read(R4/"gene_modules_display.csv")
    cells=read(R4/"input/cells.csv")
    scores=read(R4/"cell_module_eigengenes.csv").set_index("r_cell_id").reindex(cells.r_cell_id)
    names=[f"M{i}" for i in range(1,9)]
    scores=(scores[names]-scores[names].mean())/scores[names].std(ddof=0)
    colors=dict(zip(names,MODCOL));mods["display_color"]=mods.module.map(colors).fillna("#AAAAAA")
    d=dict(cells=cells,scores=scores.reset_index(drop=True),names=names,colors=colors,
           power=read(R4/"soft_power.csv"))
    original_letter=module.letter if hasattr(module,"letter") else None
    module.panel_letter=lambda *args:None
    module.letter=lambda *args:None
    panels={}
    fig,axs=plt.subplots(2,2,figsize=(4.4,3.7));fig.subplots_adjust(left=.17,right=.98,bottom=.14,top=.95,hspace=.60,wspace=.65)
    beta=int(d["power"].loc[d["power"].selected,"Power"].iloc[0]);p=d["power"]
    for k,(ax,col,lab) in enumerate(zip(axs.ravel(),["SFT.R.sq","mean.k.","median.k.","max.k."],
                                      ["Scale-free fit (R2)","Mean connectivity","Median connectivity","Max connectivity"])):
        values=p[col].to_numpy()
        if k==0:values=-np.sign(p.slope.to_numpy())*values
        ax.plot(p.Power,values,"o-",ms=2.6,lw=.65,color=BLUE)
        if k==0:ax.axhline(.8,color="#888888",lw=.7,ls="--")
        ax.axvline(beta,color=RED,lw=.8,ls="--")
        v=values[p.selected.to_numpy()][0]
        ax.scatter([beta],[v],c=RED,s=21,zorder=3)
        ax.annotate(f"beta = {beta}",(beta,v),xytext=(8,9),textcoords="offset points",fontsize=8,color=RED)
        ax.set_xlabel("Soft power",fontsize=8.5);ax.set_ylabel(lab,fontsize=8.5)
        ax.set_xticks([0,10,20,30]);ax.tick_params(labelsize=8);style(ax)
    panels["A"]=export(fig,"Fig4A_soft_power")
    fig=plt.figure(figsize=(7.2,3.7))
    xy=cells[["umap_1","umap_2"]].to_numpy()
    for k,m in enumerate(names):
        col=k%4;row=k//4
        ax=fig.add_axes([.025+col*.242,.08+(1-row)*.46,.205,.37])
        values=scores[m].to_numpy();lo,hi=np.quantile(values,[.02,.98]);order=np.argsort(values)
        cmap=LinearSegmentedColormap.from_list(m,["#D5D5D5","#FAFAFA",colors[m]])
        ax.scatter(xy[order,0],xy[order,1],c=values[order],s=1.4,vmin=lo,vmax=hi,cmap=cmap,lw=0,rasterized=True)
        ax.set_aspect("equal");ax.axis("off");ax.set_title(m,fontsize=10,fontweight="bold",pad=3)
        cax=ax.inset_axes([1.02,.17,.045,.56]);cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(lo,hi),cmap=cmap),cax=cax)
        cb.set_ticks([lo,hi]);cb.set_ticklabels(["Low","High"]);cb.ax.tick_params(labelsize=6,length=0,pad=1);cb.outline.set_visible(False)
    panels["B"]=export(fig,"Fig4B_module_maps")
    um=read(R4/"module_umap.csv")
    um=um[["gene","UMAP1","UMAP2"]].merge(mods,left_on="gene",right_on="gene_name")
    edges=read(R4/"global_network_edges.csv");positions=um.set_index("gene")[["UMAP1","UMAP2"]]
    pair=np.stack([positions.reindex(edges.source).to_numpy(),positions.reindex(edges.target).to_numpy()],axis=1)
    fig,ax=plt.subplots(figsize=(5.4,4.15));fig.subplots_adjust(left=.05,right=.93,bottom=.035,top=.92)
    ax.add_collection(LineCollection(pair,colors="#ACB5BA",lw=.2,alpha=.10,zorder=0))
    for m in names:
        u=um[um.module.eq(m)]
        ax.scatter(u.UMAP1,u.UMAP2,s=2+7*np.maximum(u.kME,0)**3,c=colors[m],lw=0,rasterized=True)
    ax.margins(.12);ax.axis("off")
    centers=um.groupby("module")[["UMAP1","UMAP2"]].median()
    for m,row in centers.iterrows():
        ax.text(row.UMAP1,row.UMAP2,m,fontsize=9,fontweight="bold",ha="center",
                 bbox=dict(facecolor="white",edgecolor="none",alpha=.88,pad=1.8))
    ax.set_title("Co-expression network | 2,134 genes",loc="left",fontsize=10,fontweight="bold")
    panels["C"]=export(fig,"Fig4C_network")
    dot=read(R4/"module_cluster_dotplot.csv")
    fig,ax=plt.subplots(figsize=(5.6,3.5));fig.subplots_adjust(left=.10,right=.77,bottom=.15,top=.89)
    xx=dot.cluster.str[1:].astype(int);yy=dot.module.str[1:].astype(int)-1
    sc=ax.scatter(xx,yy,c=dot.scaled_mean_ME,s=5+dot.percent_positive_ME*1.1,cmap="RdBu_r",vmin=-2,vmax=2,lw=.2,edgecolor="#777777")
    ax.set_xticks(range(9));ax.set_xticklabels([f"C{i}" for i in range(9)],fontsize=9)
    ax.set_yticks(range(8));ax.set_yticklabels(names,fontsize=9)
    ax.set_xlim(-.5,8.5);ax.set_ylim(-.5,7.5);ax.tick_params(length=0,pad=3)
    ax.set_xticks(np.arange(-.5,9),minor=True);ax.set_yticks(np.arange(-.5,8),minor=True)
    ax.grid(which="minor",color="#DFE3E5",lw=.5);ax.tick_params(which="minor",length=0);ax.set_axisbelow(True)
    ax.set_title("Module activity across epithelial clusters",fontsize=10,fontweight="bold",loc="left",pad=9)
    cb=fig.colorbar(sc,cax=fig.add_axes([.82,.55,.024,.25]),ticks=[-2,0,2]);cb.ax.tick_params(labelsize=8,length=0);cb.outline.set_visible(False)
    fig.text(.81,.845,"Mean ME (z)",fontsize=8)
    for v in [25,50,75]:ax.scatter([],[],s=5+1.1*v,c=INK,label=str(v))
    ax.legend(title="Positive ME (%)",loc="lower left",bbox_to_anchor=(1.06,-.02),frameon=False,fontsize=8,title_fontsize=8,labelspacing=.8)
    panels["D"]=export(fig,"Fig4D_module_activity")
    selected=read(R4/"representative_module_selection.csv").module.tolist()
    edges=read(R4/"selected_hub_network_edges.csv")
    fig,axs=plt.subplots(2,2,figsize=(6.0,5.7));fig.subplots_adjust(left=.045,right=.97,bottom=.045,top=.94,wspace=.13,hspace=.20)
    for ax,m in zip(axs.ravel(),selected):
        nodes=mods[mods.module.eq(m)].sort_values("hub_rank").head(25)
        inner=np.linspace(np.pi/2,np.pi/2+2*np.pi,5,endpoint=False)
        outer=np.linspace(np.pi/2,np.pi/2+2*np.pi,20,endpoint=False)
        xy=np.vstack([.42*np.c_[np.cos(inner),np.sin(inner)],np.c_[np.cos(outer),np.sin(outer)]])
        lookup=dict(zip(nodes.gene_name,xy));es=edges[edges.module.eq(m)]
        seg=np.array([[lookup[r.source],lookup[r.target]] for r in es.itertuples()])
        norm=(es.weight_TOM-es.weight_TOM.min())/max(es.weight_TOM.max()-es.weight_TOM.min(),1e-9)
        # Every existing edge is retained; only opacity and thickness encode strength.
        ax.add_collection(LineCollection(seg,colors=colors[m],alpha=.16,linewidths=.15+.7*norm,zorder=0))
        ax.scatter(xy[:,0],xy[:,1],s=14+25*np.maximum(nodes.kME,0),c=colors[m],edgecolor="white",lw=.4,zorder=2)
        labels=[]
        for index,(g,(x,y)) in enumerate(zip(nodes.gene_name,xy)):
            labels.append(ax.text(x*1.14,y*1.14,g,fontsize=7.5,fontstyle="italic",ha="center",va="center"))
        ax.set(xlim=(-1.55,1.55),ylim=(-1.37,1.37));ax.set_aspect("equal");ax.axis("off")
        ax.set_title(m,fontsize=11,fontweight="bold",pad=2)
        separate_network_labels(ax,labels)
    panels["E"]=export(fig,"Fig4E_hub_networks")
    go=read(R4/"GO_BP_2023_display.csv")
    fig,ax=plt.subplots(figsize=(6.4,5.7));fig.subplots_adjust(left=.63,right=.96,bottom=.105,top=.87)
    y=np.arange(len(go))[::-1];x=go.module.str[1:].astype(int)-1
    ax.scatter(x,y,s=8+np.minimum(go.minus_log10_fdr,12)*5,c=go.module.map(colors),lw=.25,edgecolor="white")
    ax.set_yticks(y);ax.set_yticklabels([wrapped(clean_go(t),38) for t in go.term],fontsize=7.6)
    ax.set_xticks(range(8));ax.set_xticklabels(names,fontsize=8)
    ax.set_xlim(-.55,7.55);ax.set_ylim(-.8,len(go)-.2);ax.tick_params(axis="y",length=0,pad=5)
    for i in range(0,len(go),6):ax.axhspan(len(go)-i-3.5,len(go)-i-.5,color="#F3F5F6",zorder=0)
    style(ax)
    ax.set_title("GO biological processes",loc="right",fontsize=10,fontweight="bold",pad=28)
    for v in [2,6,10]:ax.scatter([],[],s=8+v*5,c=INK,label=str(v))
    ax.legend(title="-log10(FDR)",ncol=3,loc="lower right",bbox_to_anchor=(1,1.01),frameon=False,
               fontsize=7,title_fontsize=7,handletextpad=.2,columnspacing=.7)
    panels["F"]=export(fig,"Fig4F_GO")
    doc=fitz.open();doc.new_page(width=864,height=965)
    put(doc,panels["A"],(16,12,336,285),"A")
    put(doc,panels["B"],(345,12,848,285),"B")
    put(doc,panels["C"],(16,295,425,583),"C")
    put(doc,panels["D"],(440,300,848,583),"D")
    put(doc,panels["E"],(16,590,430,950),"E")
    put(doc,panels["F"],(431,590,848,950),"F")
    save_doc(doc,"Figure4_visual_revision")


def figure5():
    frozen=jsonread(R5/"model_frozen_before_external_evaluation.json")
    metrics=jsonread(R5/"survival_metrics.json")
    rocs=jsonread(R5/"time_dependent_ROC_coordinates.json")
    dynamics=read(R5/"time_dependent_AUC_audit.csv")
    frames={"TCGA-OSCC":read(R5/"TCGA_nested_OOF_predictions.csv"),
             "GSE41613":read(R5/"GSE41613_frozen_predictions.csv"),
             "GSE65858-OSCC":read(R5/"GSE65858-OSCC_frozen_predictions.csv")}
    frames["External pooled"]=pd.concat([frames["GSE41613"],frames["GSE65858-OSCC"]],ignore_index=True)
    for frame in frames.values():frame["high"]=frame.risk_score>frame.groupby("cohort").risk_score.transform("median")
    risk_table=[]
    table=read(R5/"Fig5A_model_comparison_values.csv").sort_values(["mean_cindex","config"],ascending=[False,True])
    display=pd.concat([table.head(20),table[table.selected]]).drop_duplicates("config").sort_values("mean_cindex",ascending=False)
    display.to_csv(PAN/"Fig5A_displayed_configuration_ids.csv",index=False)
    fullmodule=import_module("f5original","scripts/49_evaluate_plot_figure5.py")
    fig=plt.figure(figsize=(6.4,15));fullmodule.panel_a(fig,[.015,.015,.97,.97],table)
    export(fig,"FigureS5_all_101_configurations",SUP)
    # Main page preserves every cohort and reported endpoint; only the ranked configuration display is abbreviated.
    fig=plt.figure(figsize=(12,14.5))
    def letter(x,y,s):fig.text(x,y,s,fontsize=15,fontweight="bold",va="top")
    letter(.025,.985,"A")
    ax=fig.add_axes([.20,.71,.235,.23])
    val=display[["mean_cindex","GSE41613","GSE65858-OSCC"]].to_numpy()
    cmap=LinearSegmentedColormap.from_list("centered_cindex",["#719BAD","#F6F6F3","#C56C5D"])
    ax.imshow(val,cmap=cmap,vmin=.35,vmax=.65,aspect="auto")
    short=[re.sub(r"Enet L1=","EN ",str(s)).replace(", a="," / ").replace("Ridge a=","Ridge ") for s in display.name]
    ax.set_yticks(range(len(display)));ax.set_yticklabels(short,fontsize=7.5)
    ax.set_xticks(range(3));ax.set_xticklabels(["TCGA CV","GSE41613","GSE65858"],fontsize=8)
    ax.xaxis.tick_top();ax.tick_params(length=0,pad=3)
    for i in range(len(display)):
        for j in range(3):ax.text(j,i,f"{val[i,j]:.3f}",ha="center",va="center",fontsize=7.3)
    for i in np.flatnonzero(display.selected.to_numpy()):ax.add_patch(Rectangle((-.5,i-.5),3,1,fill=False,edgecolor=INK,lw=1.3))
    ax.set_title("Top 20 TCGA-CV configurations + selected",fontsize=9.5,fontweight="bold",pad=25)
    # Exact numerical mapping is visible under the matrix; the midpoint is chance concordance.
    cb=fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(.35,.65),cmap=cmap),cax=fig.add_axes([.22,.672,.15,.010]),orientation="horizontal")
    cb.set_ticks([.35,.5,.65]);cb.ax.tick_params(labelsize=7,length=0,pad=2);cb.outline.set_visible(False)
    fig.text(.38,.666,"C-index",fontsize=7)
    letter(.50,.985,"B")
    coef=read(R5/"final_model_coefficients.csv");coef=coef[coef.nonzero].sort_values("coefficient")
    bx=fig.add_axes([.60,.855,.34,.075])
    bx.barh(range(len(coef)),coef.coefficient,height=.34,color=RED)
    bx.set_yticks(range(len(coef)));bx.set_yticklabels(coef.gene,fontsize=10,fontstyle="italic")
    bx.set_xlabel("Standardized coefficient",fontsize=8);bx.set_xlim(0,.03);bx.set_xticks([0,.01,.02,.03]);style(bx,True)
    bx.set_title("Frozen elastic-net Cox model",loc="left",fontsize=10,fontweight="bold",pad=13)
    for i,v in enumerate(coef.coefficient):bx.text(v+.0005,i,f"{v:.4f}",fontsize=8,va="center")
    forest=fig.add_axes([.64,.70,.27,.105])
    keys=["TCGA-OSCC","GSE41613","GSE65858-OSCC"]
    names=["TCGA nested CV","GSE41613","GSE65858"]
    for i,k in enumerate(keys):
        m=metrics[k];v=m["cindex"];lo,hi=m["cindex_CI"]
        forest.errorbar(v,2-i,xerr=[[v-lo],[hi-v]],fmt="o",ms=4,color=BLUE,elinewidth=1,capsize=2)
        forest.text(.675,2-i,f"{v:.3f}",fontsize=8,va="center")
    forest.axvline(.5,ls="--",color="#999999",lw=.8)
    forest.set_yticks([2,1,0]);forest.set_yticklabels(names,fontsize=8);forest.set_ylim(-.5,2.5)
    forest.set_xlim(.35,.71);forest.set_xticks([.4,.5,.6]);forest.set_xlabel("C-index (95% CI)",fontsize=8);style(forest)
    for label,cohort,top in zip("CDEF",["External pooled","TCGA-OSCC","GSE41613","GSE65858-OSCC"],[.635,.48,.325,.17]):
        letter(.025,top+.014,label)
        frame=frames[cohort];m=metrics[cohort]
        title={"External pooled":"External cohorts, stratified","TCGA-OSCC":"TCGA-OSCC, nested CV",
                "GSE41613":"GSE41613","GSE65858-OSCC":"GSE65858, oral primary"}[cohort]
        fig.text(.085,top+.008,f"{title} | n = {len(frame)}",fontsize=9,fontweight="bold")
        ay=top-.105
        km_ax=fig.add_axes([.085,ay,.255,.090])
        max_time=min(float(frame.time_months.max()),120)
        for group,color,name in [(False,GREEN,"Low"),(True,RED,"High")]:
            sub=frame[frame.high.eq(group)]
            km=KaplanMeierFitter().fit(sub.time_months,sub.event)
            times=km.survival_function_.index.to_numpy();value=km.survival_function_.iloc[:,0].to_numpy()
            ci=km.confidence_interval_.to_numpy()
            km_ax.fill_between(times,ci[:,0],ci[:,1],step="post",color=color,alpha=.10,lw=0)
            km_ax.step(times,value,where="post",color=color,lw=1.2,label=name)
            cens=sub.loc[sub.event.eq(0),"time_months"].to_numpy()
            if len(cens):km_ax.plot(cens,km.predict(cens),"|",color=color,ms=3,mew=.6)
        km_ax.set(xlim=(0,max_time),ylim=(0,1.02));km_ax.set_yticks([0,.5,1]);km_ax.set_ylabel("OS probability",fontsize=8)
        km_ax.tick_params(labelsize=7,pad=2,length=2);style(km_ax)
        km_ax.legend(loc="upper right",frameon=False,ncol=2,fontsize=6.7,handlelength=1,columnspacing=.6,borderpad=0)
        km_ax.text(.025,.08,f"HR {m['HR']:.2f} ({m['HR_low']:.2f}-{m['HR_high']:.2f})\nP = {m['pvalue']:.3f}",
                    transform=km_ax.transAxes,fontsize=7,linespacing=1.6)
        ticks=[0,24,48,72] if max_time>=72 else [0,12,36,60]
        ticks=[t for t in ticks if t<=max_time];km_ax.set_xticks(ticks)
        # Risk table uses original follow-up times, not a newly estimated model.
        risk=fig.add_axes([.085,ay-.036,.255,.022]);risk.axis("off");risk.set_xlim(0,max_time);risk.set_ylim(-.5,1.5)
        for y,group,color in [(1,False,GREEN),(0,True,RED)]:
            sub=frame[frame.high.eq(group)]
            for t in ticks:
                n=int(sub.time_months.ge(t).sum())
                risk.text(t,y,str(n),fontsize=6.6,ha="center",va="center",color=color)
                risk_table.append(dict(cohort=cohort,group="High" if group else "Low",time_months=t,n_at_risk=n))
        risk.text(-.13*max_time,1,"Low",fontsize=6.6,color=GREEN,ha="right",va="center")
        risk.text(-.13*max_time,0,"High",fontsize=6.6,color=RED,ha="right",va="center")
        if label=="F":fig.text(.172,ay-.047,"Time (months); number at risk",fontsize=7.3,ha="center")
        roc=fig.add_axes([.43,ay,.21,.090]);roc.plot([0,1],[0,1],ls="--",color="#A0A0A0",lw=.7)
        for t,color in [(12,"#B4543F"),(36,"#80609A"),(60,"#3F8872")]:
            r=rocs[cohort][str(t)]
            if r["estimable"]:roc.plot(r["fpr"],r["tpr"],color=color,lw=1.1,label=f"{t//12}y: {r['auc']:.3f}")
        roc.set(xlim=(0,1),ylim=(0,1.02));roc.set_xticks([0,.5,1]);roc.set_yticks([0,.5,1])
        roc.set_xlabel("1 - Specificity",fontsize=8,labelpad=2);roc.set_ylabel("Sensitivity",fontsize=8,labelpad=2)
        roc.tick_params(labelsize=7,length=2,pad=2);style(roc)
        roc.legend(frameon=False,loc="lower right",fontsize=6.6,labelspacing=.65,handlelength=1.2,borderpad=.1)
        if not rocs[cohort]["60"]["estimable"]:fig.text(.43,ay-.044,"5y: insufficient follow-up",fontsize=6.6,color="#666666")
        dyn=fig.add_axes([.735,ay,.23,.090]);dyn.axhline(.5,color="#C1C1C1",ls="--",lw=.65)
        for column,color,ls in [("risk_score",RED,"-"),("Age","#555555","--"),("Stage",BLUE,"-"),("Sex",GREEN,":")]:
            if cohort=="External pooled":
                one=dynamics[(dynamics.cohort=="GSE41613")&(dynamics.score==column)&dynamics.estimable].set_index("time_months").auc
                two=dynamics[(dynamics.cohort=="GSE65858-OSCC")&(dynamics.score==column)&dynamics.estimable].set_index("time_months").auc
                common=one.index.intersection(two.index);x=common;v=(97*one.loc[common]+78*two.loc[common])/175
            else:
                sub=dynamics[(dynamics.cohort==cohort)&(dynamics.score==column)&dynamics.estimable]
                x=sub.time_months;v=sub.auc
            dyn.plot(x,v,color=color,ls=ls,lw=1.3 if column=="risk_score" else .85,label="Signature" if column=="risk_score" else column)
        dyn.set(xlim=(6,72),ylim=(0,1));dyn.set_xticks([12,36,60]);dyn.set_yticks([0,.5,1])
        dyn.set_xlabel("Time (months)",fontsize=8,labelpad=2);dyn.set_ylabel("AUC(t)",fontsize=8,labelpad=2)
        dyn.tick_params(labelsize=7,length=2,pad=2);style(dyn)
        if label=="C":dyn.legend(frameon=False,loc="upper left",bbox_to_anchor=(0,1.30),ncol=4,fontsize=6.2,handlelength=1,columnspacing=.7)
    # Main-figure typography stays >=5 pt when scaled from 304.8 to 183 mm width.
    from matplotlib.text import Text
    fig.canvas.draw()
    for artist in fig.findobj(Text):
        if artist.get_text():artist.set_fontsize(max(artist.get_fontsize(),8.4))
    pd.DataFrame(risk_table).to_csv(PAN/"Fig5_risk_table_values.csv",index=False)
    path=export(fig,"Figure5_visual_revision",OUT)
    # A larger row-by-row companion keeps every curve and annotation easy to inspect.
    detail=fitz.open()
    for name in ["Fig5C_survival_and_ROC.pdf","Fig5D_survival_and_ROC.pdf","Fig5E_survival_and_ROC.pdf","Fig5F_survival_and_ROC.pdf"]:
        src=fitz.open(track(BASE/"figure5_survival"/name));detail.insert_pdf(src);src.close()
    detail.save(SUP/"FigureS5_survival_details.pdf",garbage=4,deflate=True);detail.close()


def finalize():
    combined=fitz.open();comparison=fitz.open();printdoc=fitz.open();reports=[]
    old=[BASE/"assembled_figures/Figure1_reference_layout.pdf",BASE/"assembled_figures/Figure2_reference_layout.pdf",
         BASE/"assembled_figures/Figure3_reference_layout.pdf",BASE/"figure4_epithelial_hdwgcna/Figure4_hdwgcna_reference_layout.pdf",
         BASE/"figure5_survival/Figure5_reference_layout.pdf"]
    for i in range(1,6):
        path=OUT/f"Figure{i}_visual_revision.pdf"
        src=fitz.open(path);combined.insert_pdf(src)
        pg=comparison.new_page(width=1224,height=900)
        prev=fitz.open(track(old[i-1]))
        pg.show_pdf_page(fitz.Rect(12,35,604,886),prev,0)
        pg.show_pdf_page(fitz.Rect(620,35,1212,886),src,0)
        pg.insert_text((15,22),f"Figure {i} | Previous",fontsize=13)
        pg.insert_text((623,22),f"Figure {i} | Visual revision",fontsize=13)
        prev.close()
        width=183/25.4*72;factor=width/src[0].rect.width
        pg=printdoc.new_page(width=width,height=src[0].rect.height*factor);pg.show_pdf_page(pg.rect,src,0)
        sizes=[];outside=[]
        for b in src[0].get_text("dict")["blocks"]:
            for line in b.get("lines",[]):
                for span in line["spans"]:
                    sizes.append(span["size"]*factor)
                    if not (src[0].rect+(-1,-1,1,1)).contains(fitz.Rect(span["bbox"])):outside.append(span["text"])
        assert not outside,(path.name,outside)
        assert all(f[2]!="Type3" for f in src[0].get_fonts())
        reports.append(dict(figure=i,width_mm=src[0].rect.width/72*25.4,
                            minimum_font_at_183mm=float(min(sizes)),median_font_at_183mm=float(np.median(sizes)),
                            text_runs_below_5pt_at_183mm=int(np.sum(np.array(sizes)<5)),
                            all_text_inside_page=True,Type3_fonts=0))
        src.close()
    combined.save(OUT/"Figures1-5_visual_revision.pdf",garbage=4,deflate=True);combined.close()
    comparison.save(OUT/"Before_after_comparison.pdf",garbage=4,deflate=True);comparison.close()
    printdoc.save(OUT/"Print_preview_183mm.pdf",garbage=4,deflate=True);printdoc.close()
    # Preserve full-size versions of every original panel, plus the revised numerical panels.
    detail=fitz.open()
    paths=[BASE/"Normal_OLK_OSCC_celltypes_reference_style.pdf"]
    paths+=sorted(BASE.glob("Fig1*.pdf"))
    paths+=sorted((BASE/"figure2").glob("Fig2*.pdf"))
    paths+=sorted((BASE/"figure3_epithelial").glob("Fig3*.pdf"))
    paths+=sorted((BASE/"figure4_epithelial_hdwgcna").glob("Fig4*.pdf"))
    paths+=sorted(PAN.glob("*.pdf"))
    detail_index=[]
    for path in paths:
        source=fitz.open(track(path))
        for p in range(len(source)):
            pg=detail.new_page(width=source[p].rect.width,height=source[p].rect.height)
            pg.show_pdf_page(pg.rect,source,p)
            detail_index.append(dict(page=len(detail),file=str(path.relative_to(ROOT))))
        source.close()
    detail.save(SUP/"All_panels_full_size.pdf",deflate=True,garbage=4);detail.close()
    (SUP/"panel_index.json").write_text(json.dumps(detail_index,indent=2),encoding="utf-8")
    changed=[p for p,h in USED.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h]
    assert not changed,changed
    (OUT/"visual_revision_audit.json").write_text(json.dumps(dict(source_files_unchanged=True,
        source_hashes=USED,composition=MANIFEST,print_size_audit=reports),indent=2),encoding="utf-8")
    print(json.dumps(reports,indent=2),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--figures",default="12345");args=parser.parse_args()
    for folder in [OUT,PAN,SUP]:folder.mkdir(parents=True,exist_ok=True)
    for i in args.figures:
        print("Refining Figure",i,flush=True)
        globals()[f"figure{i}"]()
    finalize()
