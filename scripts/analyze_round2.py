import json,glob,os,collections,statistics
from scipy import stats
import numpy as np

def groups(task):
    g=collections.defaultdict(dict)
    for p in glob.glob('evidence/autosota_runs_ms/'+task+'/*_s*'):
        b=os.path.basename(p); h,_,s=b.rpartition('_s')
        rj=os.path.join(p,'run','result.json')
        if not os.path.exists(rj): continue
        try: d=json.load(open(rj))
        except Exception: continue
        g[h][int(s)]=d.get('checkpoint_restored_validation_loss', d.get('best',{}).get('validation_loss'))
    return g

BASE={'joint':0.10900132196105045,'IV':0.08617336787283421,'Tg':0.1072436872066236}
# all 5 candidate iteration means, and which one was ACCEPTED
CAND={'joint':[0.092682137761427,4.588873769925988,0.09614193050757698,0.09499517374712488,0.08745355839314668],
      'IV':[0.14702206961810588,0.10239292290061712,0.11505142226815224,0.09090976044535637,0.08330403622239828],
      'Tg':[0.11032798637946446,0.10387683331089861,0.10756984477241834,0.14993594207015692,0.11754706282826032]}
ACCEPTED={'joint':0.092682137761427,'IV':0.08330403622239828,'Tg':0.10387683331089861}
LABEL={'joint':'IDEA-001 glycol-branch bugfix','IV':'IDEA-019 SAGPooling 0.5->0.7','Tg':'IDEA-003 gate-bias init 1.5'}

rng=np.random.default_rng(0); res={}
for task in ['joint','IV','Tg']:
    g=groups(task)
    f=lambda m: next((sd for h,sd in g.items() if abs(statistics.mean(sd.values())-m)<1e-9), None)
    B=f(BASE[task]); S=f(ACCEPTED[task])
    seeds=sorted(set(B)&set(S))
    b=np.array([B[s] for s in seeds]); s_=np.array([S[s] for s in seeds]); d=s_-b; n=len(seeds)
    p=float(stats.wilcoxon(s_,b,method='exact' if n<=25 else 'auto').pvalue)
    tp=float(stats.ttest_rel(s_,b).pvalue)
    idx=rng.integers(0,n,(20000,n))
    boot=100*(b[idx].mean(1)-s_[idx].mean(1))/b[idx].mean(1)
    lo,hi=np.percentile(boot,[2.5,97.5])
    # family of 5 candidates on the SAME seeds
    cand=[f(m) for m in CAND[task]]
    D=np.array([[c[x] for x in seeds] for c in cand])-b      # (k,n)
    k=D.shape[0]
    sel=int(np.argmin([abs(np.mean([c[x] for x in seeds])-ACCEPTED[task]) for c in cand]))
    tobs=np.abs(D.mean(1)/(D.std(1,ddof=1)/np.sqrt(n)))
    t_sel=tobs[sel]
    sg=rng.choice([-1.0,1.0],size=(20000,n))
    Dp=D[None,:,:]*sg[:,None,:]
    tnull=np.abs(Dp.mean(2)/(Dp.std(2,ddof=1)/np.sqrt(n))).max(1)
    p_sel=float(((tnull>=t_sel).sum()+1)/(20001))
    sd_d=float(d.std(ddof=1)); mdes=100*(1.96+0.84)*sd_d/np.sqrt(n)/b.mean()
    # TOST equivalence at +-10% / +-20% of baseline mean
    tost={}
    for bnd in (0.10,0.20):
        delta=bnd*b.mean(); se=sd_d/np.sqrt(n); df=n-1
        t1=(d.mean()+delta)/se; t2=(d.mean()-delta)/se
        tost[int(bnd*100)]=float(max(stats.t.sf(t1,df), stats.t.cdf(t2,df)))
    res[task]=dict(label=LABEL[task],n=n,base=float(b.mean()),best=float(s_.mean()),
        rel=float(100*(b.mean()-s_.mean())/b.mean()),ci_lo=float(lo),ci_hi=float(hi),
        impr=int((d<0).sum()),wilcoxon_p=p,ttest_p=tp,k_ideas=k,p_selection_adj=p_sel,
        sd_paired_diff=sd_d,mdes_pct=float(mdes),tost_p=tost,seeds=seeds)
    print('%-6s %-30s n=%2d rel=%6.2f%% CI=[%6.2f,%6.2f] impr=%2d/%2d raw_p=%.4f sel_p=%.4f MDES=%.1f%% TOST10=%.3f TOST20=%.3f'%(
        task,LABEL[task],n,res[task]['rel'],lo,hi,res[task]['impr'],n,p,p_sel,mdes,tost[10],tost[20]))

m=3
print(chr(10)+'Holm-Bonferroni across 3 tasks:')
for nm,key in (('raw Wilcoxon','wilcoxon_p'),('selection-adjusted','p_selection_adj')):
    print('  ['+nm+']')
    for i,(p,t) in enumerate(sorted((res[t][key],t) for t in res)):
        print('    %-6s p=%.4f  holm=%.4f  %s'%(t,p,min(1,p*(m-i)),'SIG' if min(1,p*(m-i))<0.05 else 'ns'))
json.dump(res,open('evidence/round2_paired_stats.json','w'),indent=2)
print(chr(10)+'saved evidence/round2_paired_stats.json')
