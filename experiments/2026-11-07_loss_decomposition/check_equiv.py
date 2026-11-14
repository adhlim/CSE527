import sys,numpy as np,pandas as pd
sys.path.insert(0,'orig_code');import features as f
B='/mnt/user-data/uploads/CSE527/experiments/2026-10-14_XFeat_quantile_8_9_10_11bin_large/results/bundle/results/'
E=np.load(B+'fixed_quantile_edges.npz')['bins_8']
def mnn_max(S):
    r=S.argmax(1);c=S.argmax(0);i=np.flatnonzero(c[r]==np.arange(len(r)));return i,r[i]
def mnn_min(D):
    r=D.argmin(1);c=D.argmin(0);i=np.flatnonzero(c[r]==np.arange(len(r)));return i,r[i]
codes=lambda x:(x[:,:,None]>E[None]).sum(2).astype(np.int16)
for label,path,csv in [('COCO_synthetic','regen/coco/test_pairs.npz','COCO_synthetic_per_pair.csv'),('HPatches','regen/hpatches_cache/hpatches_pairs.npz','HPatches_per_pair.csv')]:
    old=pd.read_csv(B+csv);pairs=f.read_pair_cache(path);rows=[]
    for p in pairs:
        a,b=p['d1'],p['d2']
        for name,(i,j) in [('official_float_cosine',mnn_max(a@b.T)),('quantile_8bin_thermometer_AVX2',mnn_min(np.abs(codes(a)[:,None]-codes(b)[None]).sum(2)))]:
            err=np.linalg.norm(f.project(p['k1'][i],p['H'])-p['k2'][j],axis=1)
            rows.append(dict(pair=p['name'],method=name,n=len(a),m=len(b),mutual=len(i),correct_3px=int((err<3).sum())))
    new=pd.DataFrame(rows);m=new.merge(old[['pair','method','n','m','mutual','correct_3px']],on=['pair','method'],suffixes=('','_old'))
    print(label,'pairs',len(m)//2)
    for meth,g in m.groupby('method'):
        print(f"  {meth}: kp n/m identical {((g.n==g.n_old)&(g.m==g.m_old)).mean():.3f} | mutual new {g.mutual.sum()} old {g.mutual_old.sum()} | correct3 new {g.correct_3px.sum()} old {g.correct_3px_old.sum()} | pairs exactly equal {((g.mutual==g.mutual_old)&(g.correct_3px==g.correct_3px_old)).mean():.3f}")
