"""Independent patient-level numerical verification from the published source tables."""
from pathlib import Path
import json, argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT=Path(__file__).resolve().parents[1]
QC=['total_counts_available','n_detected_available','pct_mt_available']
COMPS={'C1_check_QC':('C1_score','C1_check',False),
       'C1_check_QC_biology':('C1_score','C1_check',True),
       'C1_M1':('C1_score','M1',False),'C1_M5':('C1_score','M5',False)}
def design(g,bio):
    a=g[QC+(['Basal_control','Cycle_control'] if bio else [])].rank().to_numpy()
    a-=a.mean(axis=0); sd=a.std(axis=0);a=a[:,sd>1e-10]/sd[sd>1e-10]
    return np.c_[np.ones(len(g)),a]
def residual(s,d):
    a=s.rank().to_numpy();return a-d@np.linalg.solve(d.T@d,d.T@a)
def bh(p):
    p=np.asarray(p); ix=np.argsort(p); v=np.minimum.accumulate((p[ix]*len(p)/np.arange(1,len(p)+1))[::-1])[::-1]
    out=np.empty(len(p));out[ix]=np.minimum(v,1);return out
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--report',type=Path);args=ap.parse_args()
    r=ROOT/'source_tables/figure5'; f=pd.read_csv(r/'replication_cell_scores.csv.gz');sig=json.loads((r/'signatures.json').read_text(encoding="utf-8"))
    assert len(f)==24000 and f.patient.nunique()==16 and f.cell_id.is_unique
    assert not set(sig['C1_score'])&set(sig['C1_check'])
    assert not set(sig['C1_full'])&(set(sig['M1'])|set(sig['M5']))
    rows=[]
    for pat,g in f.groupby('patient'):
        for comp,(x,y,bio) in COMPS.items():
            d=design(g,bio);a=residual(g[x],d);b=residual(g[y],d)
            rows.append(dict(patient=pat,comparison=comp,rho=float(np.corrcoef(a,b)[0,1]),n_cells=len(g)))
    mine=pd.DataFrame(rows);saved=pd.read_csv(r/'replication_patient_effects.csv')
    joined=mine.merge(saved,on=['patient','comparison'],suffixes=('_audit','_saved'),validate='one_to_one')
    err=float(abs(joined.rho_audit-joined.rho_saved).max());assert len(joined)==64 and err<1e-9
    summary=pd.read_csv(r/'replication_summary.csv').set_index('comparison'); ps=[];keys=[]
    for name,g in mine.groupby('comparison'):
        v=g.rho.to_numpy(); row=summary.loc[name]
        p=binomtest(int((v>0).sum()),int((v!=0).sum()),.5).pvalue
        assert np.isclose(np.median(v),row['median'],atol=1e-9)
        assert int((v>0).sum())==row['n_positive'] and np.isclose(p,row['p'],atol=1e-12)
        ps.append(p);keys.append(name)
    assert np.allclose(bh(ps),summary.loc[keys,'bh_q'],atol=1e-12)
    sens=pd.read_csv(r/'replication_min100_sensitivity.csv').set_index('comparison')
    for name,g in mine[mine.n_cells>=100].groupby('comparison'):
        assert len(g)==sens.loc[name,'n_patients'] and np.isclose(g.rho.median(),sens.loc[name,'median'],atol=1e-9)
    null=pd.read_csv(r/'replication_matched_null_correlations.csv')
    assert len(null)==1600 and null.groupby('null_signature').patient.nunique().eq(16).all()
    med=null.groupby('null_signature').rho.median();obs=mine[mine.comparison=='C1_check_QC_biology'].rho.median()
    assert np.isclose(med.median(),.083889,atol=1e-6) and np.isclose(med.quantile(.95),.159591,atol=1e-6)
    assert (1+(med>=obs).sum())/(1+len(med))==1/101
    report={'status':'passed','n_cells':len(f),'n_patients':16,'max_patient_rho_absolute_error':err,
     'checks':['fixed disjoint signatures','64 rank-residual correlations','medians/sign tests/BH q','min100 sensitivity','100-set stored null aggregation'],
     'scope':'Independent arithmetic verification from frozen source tables. Null per-cell expression and raw preprocessing are not rerun by this script.'}
    if args.report:args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
