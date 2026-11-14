"""Merge SIFT per-pair parts into summary / by_type tables. usage: python summarize.py RESULTS_DIR DESC"""
import sys, glob
from pathlib import Path
import pandas as pd
OUT, DESC = Path(sys.argv[1]), sys.argv[2]
df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / f'per_pair_{DESC}_*_*.csv.gz')))])
keys = ['dataset', 'descriptor', 'family', 'dims', 'strategy', 'levels', 'distance', 'bytes']
assert not df.duplicated(['pair'] + keys).any(); print('pairs', df.groupby('dataset').pair.nunique().to_dict())
def summ(g):
    r = dict(pairs=len(g), mutual=int(g.mutual.sum()), correct_3px=int(g.correct_3px.sum()))
    for px in (1, 3, 5): r[f'precision_{px}px'] = 100 * g[f'correct_{px}px'].sum() / g.mutual.sum()
    return r
s = pd.DataFrame([dict(zip(keys, k), **summ(g)) for k, g in df.groupby(keys, sort=False)]); s.to_csv(OUT / f'summary_{DESC}.csv', index=False)
k2 = ['dataset', 'category'] + keys[1:]
pd.DataFrame([dict(zip(k2, k), **summ(g)) for k, g in df.groupby(k2, sort=False)]).to_csv(OUT / f'by_type_{DESC}.csv', index=False)
pd.set_option('display.width', 220)
for ds in ('HPatches', 'COCO_synthetic'):
    d = s[s.dataset == ds]; print(f"\n== {ds} [{DESC}]")
    print(d[d.family.isin(['float', 'native_u8', 'pca_float'])][['family', 'dims', 'bytes', 'mutual', 'precision_3px']].round(2).to_string(index=False))
    for dist in ('idx_L1', 'idx_L2', 'recon_cos'):
        print('--', dist); print(d[(d.family == 'scalar') & (d.distance == dist)].pivot_table(index='strategy', columns='levels', values='precision_3px').round(2).reindex(['uniform', 'percentile', 'quantile', 'lloyd', 'uniform_shared', 'quantile_shared']).to_string())
    print('-- pca + quantile'); print(d[d.family == 'pca'].pivot_table(index=['dims', 'levels', 'bytes'], columns='distance', values='precision_3px').round(2).to_string())
