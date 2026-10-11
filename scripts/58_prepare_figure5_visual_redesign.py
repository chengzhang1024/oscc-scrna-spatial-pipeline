"""New display-only coordinates and aggregates; frozen Figure 5 statistics unchanged."""
from pathlib import Path
import os
ROOT=Path(__file__).resolve().parents[1]
os.environ.setdefault('NUMBA_CACHE_DIR',str(ROOT/'.cache/numba_cache'))
os.environ.setdefault('OMP_NUM_THREADS','4')
os.environ.setdefault('OPENBLAS_NUM_THREADS','4')
import importlib.util
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import sparse
from patsy import dmatrix
import anndata as ad
import networkx as nx
import scanpy as sc

RES=ROOT/'results/paper9/figure5_program_replication'
OUT=RES/'visual_redesign'
SEED=20260916

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def standard(x):
    x=np.asarray(x,float); x=x-x.mean(axis=0)
    sd=x.std(axis=0); return x/np.where(sd>1e-10,sd,1)

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    frozen=[RES/p for p in ['signatures.json','replication_cell_scores.csv.gz','replication_patient_effects.csv','replication_summary.csv','spatial_spot_scores.csv.gz','spatial_summary.csv']]
    hashes={p.name:sha(p) for p in frozen}
    scores=pd.read_csv(RES/'replication_cell_scores.csv.gz')
    signatures=json.loads((RES/'signatures.json').read_text(encoding="utf-8"))
    spec=importlib.util.spec_from_file_location('analysis56',ROOT/'scripts/56_figure5_program_replication.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    cache=OUT/'epithelial_log_expression.npz'
    if cache.exists():
        x=sparse.load_npz(cache);genes=pd.read_csv(OUT/'gene_universe.csv').gene.to_numpy()
    else:
        x,qc,genes=module.extract_sc(scores.cell_id)
        assert np.allclose(qc[module.QC],scores[module.QC],rtol=1e-5)
        sparse.save_npz(cache,x)
        pd.DataFrame({'gene':genes}).to_csv(OUT/'gene_universe.csv',index=False)
    coords=OUT/'epithelial_umap.csv.gz'
    if not coords.exists():
        obj=ad.AnnData(x.copy(),obs=scores[['cell_id','patient']].set_index('cell_id'),var=pd.DataFrame(index=genes))
        obj.obs['patient']=obj.obs.patient.astype('category')
        sc.pp.highly_variable_genes(obj,n_top_genes=3000,batch_key='patient',flavor='seurat')
        technical=pd.Series(genes).str.match(r'^(MT-|RPL|RPS|IGH|IGK|IGL|TRAV|TRBV)').to_numpy()
        chosen=obj.var.highly_variable.to_numpy()&~technical
        pd.DataFrame({'gene':genes[chosen]}).to_csv(OUT/'embedding_hvg.csv',index=False)
        small=obj[:,chosen].copy()
        sc.pp.scale(small,max_value=10)
        sc.tl.pca(small,n_comps=30,svd_solver='arpack',random_state=SEED)
        sc.pp.neighbors(small,n_neighbors=30,n_pcs=30,random_state=SEED)
        sc.tl.umap(small,min_dist=.4,random_state=SEED)
        graph=nx.from_scipy_sparse_array(small.obsp['connectivities'])
        communities=nx.community.louvain_communities(graph,resolution=.6,seed=SEED)
        communities=sorted(communities,key=lambda s:(-len(s),min(s)))
        labels=np.empty(len(scores),object)
        for k,group in enumerate(communities):labels[list(group)]=f'E{k}'
        result=scores[['cell_id','patient']].copy()
        result[['UMAP1','UMAP2']]=small.obsm['X_umap'];result['display_cluster']=labels
        result.to_csv(coords,index=False)
        print('UMAP',len(scores),'cells',len(communities),'display clusters',flush=True)
    emb=pd.read_csv(coords)
    assert emb.cell_id.equals(scores.cell_id)
    pd.crosstab(emb.display_cluster,emb.patient).to_csv(OUT/'display_cluster_patient_counts.csv')
    # True count-summed patient pseudobulk, normalized over same shared gene universe.
    gg=signatures['C1_check'];ix=pd.Index(genes).get_indexer(gg)
    counts=x[:,ix].toarray().astype(float)
    counts=np.expm1(counts)*scores.total_counts_available.to_numpy()[:,None]/1e4
    order=scores.groupby('patient').C1_score.mean().sort_values(kind='stable').index
    pseudobulk=[]
    for patient in order:
        mask=scores.patient.eq(patient).to_numpy()
        pseudobulk.append(np.log1p(counts[mask].sum(axis=0)/scores.loc[mask,'total_counts_available'].sum()*1e6))
    expr=pd.DataFrame(np.array(pseudobulk).T,index=gg,columns=order)
    expr.index.name='gene';expr.to_csv(OUT/'checking_genes_patient_pseudobulk_logCPM.csv')
    z=pd.DataFrame(standard(expr.to_numpy().T).T,index=gg,columns=order);z.index.name='gene'
    z.to_csv(OUT/'checking_genes_patient_pseudobulk_z.csv')
    activity=scores.groupby('patient')[['C1_score','M1','M5']].mean().reindex(order)
    activity.to_csv(OUT/'patient_program_means.csv')
    pd.DataFrame(standard(activity),index=order,columns=activity.columns).to_csv(OUT/'patient_program_z.csv')
    # Section-relative residual pattern, distinct from original pairwise common-support tests.
    spots=pd.read_csv(RES/'spatial_spot_scores.csv.gz');rows=[];audit=[]
    for sample,f in spots.groupby('sample_id',sort=False):
        spline=np.asarray(dmatrix('bs(v,df=4,degree=3,include_intercept=False)-1',{'v':np.log1p(f.epithelial_abundance.to_numpy())}))
        cov=np.c_[spline,np.log1p(f.total_counts_available),np.log1p(f.n_detected_available),f.pct_mt_available]
        design=np.c_[np.ones(len(f)),standard(cov)]
        for program in ['C1_score','M1','M5']:
            y=standard(f[program].to_numpy())
            residual=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
            assert abs(residual.mean())<1e-8
            for niche in ['N1','N2','N3']:
                mask=f.niche_id.eq(niche).to_numpy()
                rows.append(dict(sample_id=sample,patient_id=f.patient_id.iloc[0],program=program,niche=niche,
                                 mean_residual=float(residual[mask].mean()) if mask.any() else np.nan,n_spots=int(mask.sum())))
        audit.append(dict(sample_id=sample,rank=int(np.linalg.matrix_rank(design)),columns=design.shape[1],condition=float(np.linalg.cond(design))))
    pd.DataFrame(rows).to_csv(OUT/'section_program_niche_residuals.csv',index=False)
    pd.DataFrame(audit).to_csv(OUT/'section_residual_design_audit.csv',index=False)
    assert all(sha(RES/name)==value for name,value in hashes.items())
    manifest=dict(seed=SEED,original_result_hashes=hashes,n_cells=len(emb),n_patients=scores.patient.nunique(),
      embedding='GSE310797 epithelial-only; patient-batched HVG 3000, technical genes excluded; scaled max10, PCA30, neighbors30, UMAP min_dist0.4; no integration/batch correction; fixed Louvain resolution0.6. No tuning to program maps.',
      cluster_interpretation='E0... are descriptive graph groups, not C1 labels or validated biological states.',
      B='All 29 frozen checking genes; true patient count-summed pseudobulk log1p CPM; row z across 16 patients, display clipped at +/-2. Patients ordered by continuous mean C1 score. Not patient-wise correlation.',
      D='Patient mean log-normalized program activity; column z across patients. Descriptive between-patient activity cannot establish within-patient coupling; see statistics supplement.',
      F='Mean section-standardized program residual in each existing niche after spline adjustment of epithelial abundance plus QC, using all spots within section. Descriptive spatial pattern; NOT the previously estimated pairwise niche coefficient. Missing niches gray; small nonzero niches retained and counts reported. No inferential P values.',
      scores='All C1/M1/M5 scores and gene sets unchanged. M1/M5 exclude frozen C1 marker overlap.',
      spatial='S6 and S2 unchanged. Same program color limits across sections; same coordinates/crop for all maps within section.',
      source_code_sha256=sha(Path(__file__)))
    (OUT/'display_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print('DISPLAY DATA READY',flush=True)

if __name__=='__main__':main()
