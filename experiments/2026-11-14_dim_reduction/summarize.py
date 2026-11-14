"""Merge per-pair parts into summary.csv / by_type.csv, and build the equal-byte comparison against the
64-D scalar-quantization results of 2026-11-08_accuracy_matrix.
usage: python summarize.py RESULTS_DIR ACCURACY_MATRIX_SUMMARY.csv"""
import sys, glob
from pathlib import Path
import pandas as pd
OUT = Path(sys.argv[1]); AM = pd.read_csv(sys.argv[2])
parts = sorted(glob.glob(str(OUT / 'per_pair_*_*.csv.gz')))
df = pd.concat([pd.read_csv(p) for p in parts])
assert not df.duplicated(['dataset', 'pair', 'method', 'dims', 'levels', 'strategy', 'distance']).any()
print('parts', len(parts), 'pairs', df.groupby('dataset').pair.nunique().to_dict())
def summ(g):
    r = dict(pairs=len(g), mutual=int(g.mutual.sum()), correct_3px=int(g.correct_3px.sum()))
    for px in (1, 3, 5): r[f'precision_{px}px'] = 100 * g[f'correct_{px}px'].sum() / g.mutual.sum()
    return r
keys = ['dataset', 'method', 'dims', 'levels', 'strategy', 'distance', 'bytes']
s = pd.DataFrame([dict(zip(keys, k), **summ(g)) for k, g in df.groupby(keys, sort=False)])
s.to_csv(OUT / 'summary.csv', index=False)
k2 = ['dataset', 'category'] + keys[1:]
pd.DataFrame([dict(zip(k2, k), **summ(g)) for k, g in df.groupby(k2, sort=False)]).to_csv(OUT / 'by_type.csv', index=False)

am = AM[AM.strategy != 'float'].copy(); am['method'] = 'scalar'; am['dims'] = 64
am = am.rename(columns={'packed_bytes': 'bytes'})
cols = ['dataset', 'method', 'dims', 'levels', 'strategy', 'distance', 'bytes', 'precision_3px']
both = pd.concat([s[s.method != 'float'][cols], am[cols]]); both['bytes'] = both['bytes'].astype(int)
both['family'] = both.apply(lambda r: 'scalar 64-D' if r.method == 'scalar' else
                            f"{r.method} float" if r.levels == 0 else f"{r.method} + quantize", axis=1)
both.to_csv(OUT / 'equal_bytes_all.csv', index=False)
# best configuration of each (family, dims) at each byte budget
best = both.sort_values('precision_3px', ascending=False).groupby(['dataset', 'bytes', 'family', 'dims'], as_index=False).first()
best['bits'] = best.levels.map({0: 32, 4: 2, 8: 3, 16: 4, 32: 5, 256: 8})
best = best[['dataset', 'bytes', 'family', 'dims', 'bits', 'strategy', 'distance', 'precision_3px']]
best.sort_values(['dataset', 'bytes', 'precision_3px'], ascending=[True, True, False]).to_csv(OUT / 'equal_bytes_best.csv', index=False)
pd.set_option('display.width', 200)
for ds in ('HPatches', 'COCO_synthetic'):
    fl = float(s[(s.dataset == ds) & (s.method == 'float')].precision_3px.iloc[0])
    print(f'\n== {ds}: float cosine (256 B) = {fl:.2f}')
    b = best[(best.dataset == ds) & (best.family != 'truncate float') & best.bytes.isin([8, 12, 16, 24, 32, 48, 64, 96, 128])]
    print(b.sort_values(['bytes', 'precision_3px'], ascending=[True, False]).drop(columns='dataset').round(2).to_string(index=False))
    g = s[(s.dataset == ds) & s.method.isin(['pca', 'gfr'])].pivot_table(index=['dims', 'levels', 'strategy', 'distance'], columns='method', values='precision_3px')
    d = (g.gfr - g.pca)
    print(f'GFR - PCA over {len(d)} configs: mean {d.mean():+.3f} pp, min {d.min():+.3f}, max {d.max():+.3f}')
    t = s[(s.dataset == ds) & (s.levels == 0) & (s.method != 'float')].pivot_table(index='dims', columns='method', values='precision_3px').round(2)
    print(t.sort_index(ascending=False).to_string())
