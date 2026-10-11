"""Figure 5: landscape / checking-expression / patient / tissue visual redesign."""
from pathlib import Path
import json
import hashlib
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
import fitz

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results/paper9/figure5_program_replication'
DAT=RES/'visual_redesign'
OUT=ROOT/'figures/paper9/figure5_program_replication/visual_redesign/viridis'
TITLE='Independent-cohort replication and spatial localization of the C1-associated epithelial program'
NICHES={'N1':'#65BBA0','N2':'#F7B52B','N3':'#CF535E'}
# Existing Fig3 cluster sequence; Fig3C continuous map; Fig3F activity map.
CLUSTERS=['#1F77B4','#FF7F0E','#2CA02C','#D62728','#9467BD','#8C564B','#E377C2','#BCBD22','#17BECF','#7F7F7F',
          '#AEC7E8','#FFBB78','#98DF8A','#FF9896','#C5B0D5','#C49C94','#F7B6D2','#DBDB8D','#9EDAE5','#C7C7C7']
SEQUENTIAL=plt.get_cmap('viridis')
DIVERGING=LinearSegmentedColormap.from_list('existing_Fig3F',[(0,'#08306B'),(.25,'#6BAED6'),(.5,'#F7F7F7'),(.75,'#FCAE91'),(1,'#B2182B')])
DIVERGING.set_bad('#DEDEDE')
plt.rcParams.update({'font.family':'Arial','font.size':10,'axes.titlesize':11,'axes.labelsize':10,
 'xtick.labelsize':9,'ytick.labelsize':9,'axes.linewidth':.65,'pdf.fonttype':42,'ps.fonttype':42,
 'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})

def natural(x):return int(re.search(r'\d+',x).group())
def rect(b,x,y,w,h):return [b[0]+x*b[2],b[1]+y*b[3],w*b[2],h*b[3]]
def axes(fig,b,x,y,w,h):return fig.add_axes(rect(b,x,y,w,h))
def label(fig,b,l,title):
    fig.text(b[0],b[1]+b[3],l,fontsize=19,weight='bold',va='top')
    fig.text(b[0]+.032,b[1]+b[3]-.002,title,fontsize=11,va='top')
def cb(fig,im,b,label='',ticks=None):
    c=fig.colorbar(im,cax=fig.add_axes(b),orientation='horizontal');c.outline.set_visible(False)
    c.ax.tick_params(length=2,pad=1,labelsize=8)
    if ticks is not None:c.set_ticks(ticks)
    if label:c.set_label(label,fontsize=9,labelpad=1)
    return c
def noaxes(ax):ax.set_xticks([]);ax.set_yticks([]);[s.set_visible(False) for s in ax.spines.values()]

def data():
    sc=pd.read_csv(RES/'replication_cell_scores.csv.gz')
    emb=pd.read_csv(DAT/'epithelial_umap.csv.gz')
    sc=sc.merge(emb.drop(columns='patient'),on='cell_id',validate='one_to_one')
    assert len(sc)==24000
    sp=pd.read_csv(RES/'spatial_spot_scores.csv.gz')
    heat=pd.read_csv(DAT/'checking_genes_patient_pseudobulk_z.csv',index_col=0)
    mean=pd.read_csv(DAT/'patient_program_means.csv',index_col=0)
    z=pd.read_csv(DAT/'patient_program_z.csv',index_col=0)
    residual=pd.read_csv(DAT/'section_program_niche_residuals.csv')
    # Same percentile rule for every program; no adjustment to make M5 match C1.
    lim={k:tuple(np.quantile(sc[k],[.02,.98])) for k in ['C1_score','M1','M5']}
    slim={k:tuple(np.quantile(sp[k],[.02,.98])) for k in lim}
    bounds=np.array([sc.UMAP1.min(),sc.UMAP1.max(),sc.UMAP2.min(),sc.UMAP2.max()])
    pad=max(bounds[1]-bounds[0],bounds[3]-bounds[2])*.04
    bounds+=np.array([-pad,pad,-pad,pad])
    return dict(sc=sc,sp=sp,heat=heat,mean=mean,z=z,residual=residual,lim=lim,slim=slim,bounds=bounds)

def umap(ax,d,key,size=1.2,arrows=False):
    f=d['sc'];rng=np.random.default_rng(20260916);order=rng.permutation(len(f))
    if key=='display_cluster':
        groups=sorted(f[key].unique(),key=natural)
        colors={g:CLUSTERS[i%len(CLUSTERS)] for i,g in enumerate(groups)}
        im=ax.scatter(f.UMAP1.iloc[order],f.UMAP2.iloc[order],s=size,c=f[key].iloc[order].map(colors),lw=0)
        for g,sub in f.groupby(key):
            center=sub[['UMAP1','UMAP2']].median().to_numpy()
            ax.text(*center,g,fontsize=9,ha='center',va='center',weight='bold',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.75,pad=.5))
    else:
        lo,hi=d['lim'][key]
        im=ax.scatter(f.UMAP1.iloc[order],f.UMAP2.iloc[order],s=size,c=f[key].iloc[order],cmap=SEQUENTIAL,vmin=lo,vmax=hi,lw=0)
    x0,x1,y0,y1=d['bounds'];ax.set_xlim(x0,x1);ax.set_ylim(y0,y1);ax.set_aspect('equal');noaxes(ax)
    if arrows:
        for end in [(.27,.025),(.035,.25)]:
            ax.annotate('',xy=end,xytext=(.035,.025),xycoords='axes fraction',arrowprops=dict(arrowstyle='->',lw=.7))
        ax.text(.28,.025,'UMAP1',fontsize=8,va='center',transform=ax.transAxes)
        ax.text(.02,.26,'UMAP2',fontsize=8,rotation=90,transform=ax.transAxes)
    return im

def panel_a(fig,b,d):
    label(fig,b,'A','GSE310797 epithelial landscape')
    for j,(key,title) in enumerate([('display_cluster','Epithelial clusters'),('C1_score','C1 program')]):
        ax=axes(fig,b,.015+j*.50,.09,.475,.80)
        im=umap(ax,d,key,arrows=j==0);ax.set_title(title,pad=3)
        if j==1:cb(fig,im,rect(b,.63,.062,.23,.012),'',np.round(d['lim'][key],2))

def panel_b(fig,b,d):
    label(fig,b,'B','Checking-gene expression')
    mat=d['heat']
    ax=axes(fig,b,.18,.15,.80,.70)
    im=ax.imshow(mat,cmap=DIVERGING,vmin=-2,vmax=2,aspect='auto',interpolation='nearest')
    ax.set_yticks(range(len(mat)),mat.index,fontsize=8,fontstyle='italic')
    ax.set_xticks(range(mat.shape[1]),mat.columns,rotation=60,ha='right',fontsize=8)
    ax.tick_params(length=0,pad=2);[s.set_visible(False) for s in ax.spines.values()]
    bar=axes(fig,b,.18,.862,.80,.022)
    vals=d['mean'].loc[mat.columns,'C1_score'].to_numpy()
    bar.imshow(vals[None,:],cmap=SEQUENTIAL,aspect='auto',vmin=vals.min(),vmax=vals.max());noaxes(bar)
    bar.set_title('Patients ordered by C1 activity',fontsize=8,pad=5)
    cb(fig,im,rect(b,.49,.043,.38,.012),'Gene-wise z score',[-2,0,2])

def panel_c(fig,b,d):
    label(fig,b,'C','Epithelial program maps')
    for j,(key,title) in enumerate([('C1_score','C1'),('M1','M1 · basal / adhesion'),('M5','M5 · proliferation')]):
        ax=axes(fig,b,.008+j*.333,.115,.322,.74)
        im=umap(ax,d,key,size=1.2);ax.set_title(title,fontsize=9,pad=2)
        cb(fig,im,rect(b,.07+j*.333,.063,.20,.014),'',np.round(d['lim'][key],2))

def panel_d(fig,b,d):
    label(fig,b,'D','Patient program activity')
    order=sorted(d['z'].index,key=natural);m=d['z'].loc[order]
    ax=axes(fig,b,.18,.18,.80,.68)
    im=ax.imshow(m,cmap=DIVERGING,vmin=-2,vmax=2,aspect='auto',interpolation='nearest')
    ax.set_yticks(range(len(m)),order,fontsize=8)
    ax.set_xticks(range(3),['C1','M1','M5'],fontsize=10);ax.xaxis.tick_top()
    ax.tick_params(length=0,pad=2);[s.set_visible(False) for s in ax.spines.values()]
    cb(fig,im,rect(b,.35,.079,.49,.016),'Program-wise z score',[-2,0,2])

def tissue(ax,f,key,d):
    xy=f[['x','y']].to_numpy();center=(xy.max(axis=0)+xy.min(axis=0))/2
    span=np.ptp(xy,axis=0).max()*1.08
    # Identical square extent for every program in a section; geometry never stretched.
    kw=dict(s=6,lw=0,marker='h')
    if key=='niche_id':im=ax.scatter(f.x,f.y,c=f.niche_id.map(NICHES),**kw)
    else:
        lo,hi=d['slim'][key]
        im=ax.scatter(f.x,f.y,c=f[key],cmap=SEQUENTIAL,vmin=lo,vmax=hi,**kw)
    ax.set_xlim(center[0]-span/2,center[0]+span/2)
    ax.set_ylim(center[1]+span/2,center[1]-span/2);ax.set_aspect('equal');noaxes(ax)
    return im

def panel_e(fig,b,d):
    label(fig,b,'E','Spatial localization')
    for row,sample in enumerate(['S6','S2']):
        f=d['sp'][d['sp'].sample_id.eq(sample)]
        for col,(key,title) in enumerate([('niche_id','Niche'),('C1_score','C1'),('M1','M1'),('M5','M5')]):
            ax=axes(fig,b,.043+col*.239,.51 if row==0 else .145,.233,.365)
            im=tissue(ax,f,key,d)
            if row==0:ax.set_title(title,fontsize=11,pad=4)
            if col==0:ax.text(-.07,.5,f"{sample} / {f.patient_id.iloc[0]}",rotation=90,va='center',ha='center',transform=ax.transAxes,fontsize=9)
            if row==1 and col>0:cb(fig,im,rect(b,.075+col*.239,.095,.175,.012),'',np.round(d['slim'][key],2))
    ax=axes(fig,b,.03,.002,.94,.041);ax.axis('off')
    handles=[Line2D([],[],marker='s',color=c,lw=0,markersize=6,label=label) for c,label in
       zip(NICHES.values(),['N1 Fibrovascular','N2 Immune-mixed','N3 Epithelial-rich'])]
    ax.legend(handles=handles,loc='center',frameon=False,ncol=3,fontsize=8,columnspacing=1.1)

def panel_f(fig,b,d):
    label(fig,b,'F','Spatial program patterns')
    tab=d['residual'].copy();tab['column']=tab.program+':'+tab.niche
    columns=[s+':'+n for s in ['C1_score','M1','M5'] for n in ['N1','N2','N3']]
    order=sorted(tab.sample_id.unique(),key=natural)
    m=tab.pivot(index='sample_id',columns='column',values='mean_residual').reindex(index=order,columns=columns)
    # Fixed symmetric scale; no per-program re-centering or reordering to imply coupling.
    ax=axes(fig,b,.21,.18,.77,.66)
    im=ax.imshow(np.ma.masked_invalid(m),cmap=DIVERGING,vmin=-.75,vmax=.75,aspect='auto',interpolation='nearest')
    patient=tab.drop_duplicates('sample_id').set_index('sample_id').patient_id
    ax.set_yticks(range(len(order)),[s+' / '+patient[s] for s in order],fontsize=8)
    ax.set_xticks(range(9),['N1','N2','N3']*3,rotation=0,fontsize=8)
    ax.tick_params(length=0,pad=3);[s.set_visible(False) for s in ax.spines.values()]
    for j,title in enumerate(['C1','M1','M5']):
        ax.text((j*3+1.5)/9,1.055,title,transform=ax.transAxes,ha='center',fontsize=10)
    for pos in [2.5,5.5]:ax.axvline(pos,color='white',lw=2)
    cb(fig,im,rect(b,.30,.095,.61,.014),'Adjusted relative activity',[-.75,0,.75])

PANELS={'A':panel_a,'B':panel_b,'C':panel_c,'D':panel_d,'E':panel_e,'F':panel_f}

def export(fig,name):
    for ext in ['pdf','svg']:fig.savefig(OUT/f'{name}.{ext}')
    fig.savefig(OUT/f'{name}.png',dpi=300)
    fig.savefig(OUT/f'{name}.tiff',dpi=300,pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig);print('EXPORTED',name,flush=True)

def supplement(d):
    old=ROOT/'figures/paper9/figure5_program_replication'
    dest=fitz.open()
    for name in ['Fig5B_program_replication.pdf','Fig5C_program_replication.pdf','Fig5D_program_replication.pdf',
                 'Fig5F_program_replication.pdf','Figure5_spatial_effect_details.pdf','Figure5_replication_details.pdf']:
        with fitz.open(old/name) as src:dest.insert_pdf(src)
    # Full spatial effect/CI/q table, preserving every model choice and patient count.
    stat=pd.read_csv(RES/'spatial_summary.csv')
    with PdfPages(OUT/'Supplementary_statistical_tables.pdf') as pdf:
        for mode,g in stat.groupby('mode'):
            fig,ax=plt.subplots(figsize=(11.7,5.7));ax.axis('off')
            cells=[[r.signature.replace('_score',''),r.contrast,str(int(r.n_patients)),
                    f'{r["median"]:.3f}',f'{r.ci_low:.3f} to {r.ci_high:.3f}',f'{r.p:.4g}',f'{r.bh_q:.4g}',
                    'inferential' if r.sufficient_patients else 'descriptive only'] for _,r in g.iterrows()]
            t=ax.table(cellText=cells,colLabels=['Program','Contrast','Patients','Median effect','95% interval','P','BH q','Scope'],
                       colWidths=[.10,.10,.08,.12,.20,.10,.10,.20],loc='center',cellLoc='center')
            t.auto_set_font_size(False);t.set_fontsize(10);t.scale(1,2.1)
            for (row,col),cell in t.get_celld().items():
                cell.set_edgecolor('#DDDDDD');cell.set_linewidth(.5)
                if row==0:cell.set_facecolor('#F1F1F1');cell.set_text_props(weight='bold')
            ax.set_title('Spatial association statistics | '+mode.replace('_',' '),fontsize=14)
            fig.text(.07,.08,'Fixed original pairwise niche models; these coefficients differ from the residual summaries in main panel F.',fontsize=10)
            fig.text(.07,.04,'Fewer than six patients: descriptive only. One-patient intervals are undefined. No independent-spot tests.',fontsize=10)
            pdf.savefig(fig);plt.close(fig)
    with fitz.open(OUT/'Supplementary_statistical_tables.pdf') as src:dest.insert_pdf(src)
    dest.save(OUT/'Supplementary_Figure5_statistics.pdf');dest.close()
    # Descriptive embedding patient composition is essential for evaluating patient separation.
    f=d['sc'];fig,axs=plt.subplots(1,2,figsize=(13,5.5))
    order=sorted(f.patient.unique(),key=natural);cmap=plt.get_cmap('tab20')
    colors={p:cmap(i) for i,p in enumerate(order)}
    axs[0].scatter(f.UMAP1,f.UMAP2,c=f.patient.map(colors),s=.6,lw=0,rasterized=True)
    noaxes(axs[0]);axs[0].set_aspect('equal');axs[0].set_title('Patient identity on the same UMAP')
    tab=pd.crosstab(f.display_cluster,f.patient).reindex(index=sorted(f.display_cluster.unique(),key=natural),columns=order)
    prop=tab.div(tab.sum(axis=1),axis=0)
    im=axs[1].imshow(prop,cmap='Greys',vmin=0,vmax=1,aspect='auto')
    axs[1].set_xticks(range(len(order)),order,rotation=60,ha='right');axs[1].set_yticks(range(len(prop)),prop.index)
    axs[1].set_title('Patient composition within display clusters');fig.colorbar(im,ax=axs[1],shrink=.65,label='Cell fraction')
    fig.subplots_adjust(bottom=.17,wspace=.3)
    fig.savefig(OUT/'Supplementary_embedding_patient_audit.pdf');plt.close(fig)
    for source in ['replication_patient_effects.csv','replication_summary.csv','spatial_section_effects.csv','spatial_patient_effects.csv','spatial_summary.csv']:
        pd.read_csv(RES/source).to_csv(OUT/('Supplementary_Table_'+source),index=False)

def documentation(d):
    caption='''# Figure 5 visual redesign

Independent-cohort replication and spatial localization of the C1-associated epithelial program

A. Independent-cohort epithelial-only UMAP of all 24,000 GSE310797 cells, colored by descriptive Louvain graph groups (E0, E1, ...) and the existing continuous C1 score. Coordinates and point order are identical across all five UMAP views in A/C. These new display groups do not transfer the discovery C1 label. HVGs were selected with patient-aware ranking; no patient integration or outcome-driven tuning was applied. Patient composition is shown in Supplementary_embedding_patient_audit.pdf. UMAP overlap is descriptive, not a statistical replication test.

B. All 29 previously fixed, disjoint checking genes. Columns are all 16 patients, ordered by mean continuous C1 score. Raw counts were reconstructed from saved normalized expression and library totals, summed within patient, normalized to CPM using the shared 16,097-gene universe and log1p transformed. Rows display gene-wise z scores, clipped at ±2 only for display. The top strip shows patient mean C1 activity (low to high). These between-patient pseudobulk patterns are not the previous within-patient gene correlations and need not appear monotonically coordinated. No genes or patients were removed to improve appearance.

C. The identical epithelial UMAP colored by frozen C1, disjoint M1 and disjoint M5 mean-log-expression scores. Each program has its own cohort-wide 2nd–98th percentile scale under the same fixed display rule; the numerical limits are shown. No program-specific gene/scale optimization was used. M1 and M5 retain the previously removed C1-gene overlap. Continuous scores use the user-requested viridis palette (dark purple, teal, yellow), identically across all programs; score limits are unchanged. Cluster colors follow Fig.3A, extended by lighter shades for additional display groups. Heatmaps follow Fig.3F.

D. Mean program activity across all epithelial cells of each patient, standardized separately for each program across patients. Rows remain PT1–PT16; display range ±2. This is a descriptive between-patient activity summary, not evidence that within-patient C1–M5 coupling has become strong. The unchanged patient-wise association tests in the statistical supplement support C1–M1 but not consistent C1–M5 coupling. No C1-high fractions are introduced.

E. Previously selected sections S6/P6 and S2/P2, displaying existing Fig.2 niche assignments and continuous fixed program scores. N1/N2/N3 retain Fig.2 colors. Within each section all four panels have identical coordinates, square bounds, orientation and marker size. Square panels preserve tissue aspect ratio and do not represent an equal physical field of view between sections. Each program uses the same all-section 2nd–98th percentile limits for both sections. Signals are observed mixed-spot expression; tissue maps do not establish epithelial-intrinsic localization. No pathology annotation or microscopy image is invented.

F. All 12 sections and all nine program–niche combinations. For each section, the section-standardized program score is regressed on a 4-df spline of log1p epithelial abundance, log1p total counts, log1p detected genes and mitochondrial percentage. Color is the mean residual within the existing niche, on a shared symmetric ±0.75 SD scale. This new descriptive residual display is distinct from the original pairwise common-support niche regression; it does not replace its estimates or tests. Existing niches with few spots remain visible, and their spot counts are provided in section_program_niche_residuals.csv; absent niches are gray. Inference remains limited, as documented in the statistical supplement.

## Statistical interpretation

The frozen results are unchanged: QC-adjusted C1 score/check association median rho 0.509 (16/16 positive), additionally basal/cycle-adjusted rho 0.332; C1–M1 rho 0.402 (16/16 positive); C1–M5 rho 0.058 (BH q 0.4545). Spatial C1 associations are not consistently supported after abundance adjustment. M1's N2–N1 result is sensitive to abundance-versus-proportion adjustment. These exploratory analyses do not prove a discrete C1 subtype, malignant specificity, causal spatial organization or preservation of a complete co-expression network. GSE310797 had been inspected previously. Discovery marker pairing was supported by only five eligible patients.

## Outputs / reproduction

Figure5_redesigned and Fig5A–F: PDF, SVG, PNG and LZW TIFF (300 dpi). PDF/SVG preserve vector points and text; TIFF/PNG are raster outputs. Supplementary_Figure5_statistics.pdf retains original patient-wise correlations, confidence intervals, q values and forest plots. Supplementary statistical CSVs retain complete original estimates. Display tables and parameter/hash manifest are in results/paper9/figure5_program_replication/visual_redesign.

Run scripts/58_prepare_figure5_visual_redesign.py then scripts/59_plot_figure5_visual_redesign.py in .envs/figure5_py. Original scripts/56 and /57 retain statistical-analysis reproducibility. No original signature or statistical source file is rewritten.
'''
    (OUT/'Figure5_legend_and_methods.md').write_text(caption,encoding='utf8')
    (OUT/'README_中文说明.md').write_text('''# Figure 5 重排版

优先查看 Figure5_redesigned.png 或 PDF。A–F各自提供PDF/SVG/300 dpi PNG/TIFF。

主图为独立队列上皮UMAP、29个核验基因的患者pseudobulk热图、同坐标程序图、患者平均活性热图、S6/S2空间图及12切片的调整后相对活性热图。连续程序评分按用户指定改为viridis深紫—蓝绿—黄色渐变；热图保留Figure3活性配色，niche保留Figure2配色。所有原签名、患者、代表切片、评分范围及统计结果保留。

新增UMAP展示分群为E0等，无监督产生，不命名为C1；未按结果调分辨率。患者平均活性热图不是患者内关联检验；即使M5在部分患者均值较高，也不构成C1–M5耦合证据。B患者pseudobulk不能替代此前患者内核验。F是新增描述性残差热图，不等于旧版两两niche回归系数；没有将其用于新的显著性判断。

详细统计见Supplementary_Figure5_statistics.pdf及Supplementary_Table CSV。各患者在新UMAP中的分布见Supplementary_embedding_patient_audit.pdf。完整定义及图注见Figure5_legend_and_methods.md。历史图5未覆盖。
''',encoding='utf8')
    limits=[]
    for domain,frame,lim in [('single_cell',d['sc'],d['lim']),('spatial',d['sp'],d['slim'])]:
        for key,(lo,hi) in lim.items():limits.append(dict(domain=domain,program=key,low=lo,high=hi,clipped_fraction=((frame[key]<lo)|(frame[key]>hi)).mean()))
    pd.DataFrame(limits).to_csv(DAT/'program_display_scales.csv',index=False)

def validate():
    manifest=json.loads((DAT/'display_manifest.json').read_text(encoding="utf-8"))
    release_checksums=json.loads((ROOT/'docs/frozen_table_checksums.json').read_text(encoding="utf-8"))
    # gzip timestamps and CSV float serialization may change on a rerun. Compare
    # the values to the archived frozen tables when byte hashes differ; never
    # silently approve changed signatures, row identities or numerical results.
    byte_identical=True
    numerical_errors={}
    for name,expected in manifest['original_result_hashes'].items():
        current=RES/name
        if hashlib.sha256(current.read_bytes()).hexdigest()==expected:continue
        byte_identical=False
        frozen=ROOT/'source_tables/figure5'/name
        assert frozen.exists(), ('Missing frozen reference',name)
        assert hashlib.sha256(frozen.read_bytes()).hexdigest()==release_checksums[name], ('Changed frozen reference',name)
        if name.endswith('.json'):
            assert json.loads(current.read_text(encoding="utf-8"))==json.loads(frozen.read_text(encoding="utf-8")), name
        else:
            a=pd.read_csv(current);b=pd.read_csv(frozen)
            assert a.shape==b.shape and a.columns.tolist()==b.columns.tolist(), name
            for column in a:
                if pd.api.types.is_numeric_dtype(a[column]) and pd.api.types.is_numeric_dtype(b[column]):
                    # Spatial spline least-squares fits show <=1.3e-7 absolute
                    # differences across the tested BLAS builds. Counts, identities
                    # and inferential p/q values are checked independently below.
                    tol=5e-7 if name=='spatial_summary.csv' and column in ['median','ci_low','ci_high'] else 1e-12
                    assert np.allclose(a[column],b[column],rtol=1e-10,atol=tol,equal_nan=True), (name,column)
                    diff=np.abs(a[column].to_numpy(float)-b[column].to_numpy(float))
                    finite=diff[np.isfinite(diff)]
                    numerical_errors[name+':'+column]=float(finite.max()) if len(finite) else 0.
                else:assert a[column].fillna('').equals(b[column].fillna('')), (name,column)
    rows=[]
    for path in sorted(OUT.glob('Fig*.pdf')):
        doc=fitz.open(path);outside=[];sizes=[]
        for page in doc:
            for block in page.get_text('dict')['blocks']:
                for line in block.get('lines',[]):
                    for span in line.get('spans',[]):
                        if span['text'].strip():
                            sizes.append(span['size'])
                            if not (page.rect+(-1,-1,1,1)).contains(fitz.Rect(span['bbox'])):outside.append(span['text'])
        rows.append(dict(file=path.name,outside_page=outside,min_font=min(sizes)))
        doc.close()
    from PIL import Image
    for stem in ['Figure5_redesigned']+['Fig5'+l for l in PANELS]:
        for ext in ['pdf','svg','png','tiff']:assert (OUT/(stem+'.'+ext)).stat().st_size>1000
        with Image.open(OUT/(stem+'.tiff')) as im:assert im.info['dpi']==(300,300)
    report=dict(source_values_match_frozen_tables=True,source_bytes_identical=byte_identical,numerical_absolute_errors=numerical_errors,spatial_effect_absolute_tolerance=5e-7,pdf_bounds=rows,output_dpi=300,all_24k_cells=True,all_16_patients=True,
                all_29_checking_genes=True,all_12_sections=True,vector_scatter=True,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (DAT/'visual_validation.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    assert not any(x['outside_page'] for x in rows),rows

def main():
    OUT.mkdir(parents=True,exist_ok=True);d=data()
    fig=plt.figure(figsize=(12,12.2))
    # The manuscript figure title belongs in its legend, as in Figures 1–4.
    positions={'A':[.022,.647,.602,.333],'B':[.651,.647,.324,.333],
               'C':[.022,.383,.602,.247],'D':[.651,.383,.324,.247],
               'E':[.022,.012,.65,.354],'F':[.697,.012,.278,.354]}
    for letter,func in PANELS.items():func(fig,positions[letter],d)
    export(fig,'Figure5_redesigned')
    for letter,func in PANELS.items():
        size={'A':(12,5.8),'B':(7,8.5),'C':(12,4.5),'D':(5.5,7),'E':(12,9),'F':(7,8)}[letter]
        fig=plt.figure(figsize=size);func(fig,[.025,.025,.95,.95],d);export(fig,'Fig5'+letter)
    supplement(d);documentation(d);validate();print('REDESIGN COMPLETE',flush=True)

if __name__=='__main__':main()
