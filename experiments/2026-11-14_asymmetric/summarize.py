"""Merge asymmetric per-pair parts; write summary.csv, by_type.csv and equal_bytes_sym_vs_asym.csv.
usage: python summarize.py   (run inside this folder)"""
import glob, pandas as pd
df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob('results/per_pair_*_*.csv.gz'))])
assert not df.duplicated(['dataset', 'pair', 'family', 'dims', 'levels', 'strategy', 'setting']).any()
keys = ['dataset', 'family', 'dims', 'levels', 'strategy', 'setting', 'bytes']
def summ(g):
    r = dict(pairs=len(g), mutual=int(g.mutual.sum()), correct_3px=int(g.correct_3px.sum()))
    for px in (1, 3, 5): r[f'precision_{px}px'] = 100 * g[f'correct_{px}px'].sum() / g.mutual.sum()
    return r
s = pd.DataFrame([dict(zip(keys, k), **summ(g)) for k, g in df.groupby(keys, sort=False)]); s.to_csv('results/summary.csv', index=False)
k2 = ['dataset', 'category'] + keys[1:]
pd.DataFrame([dict(zip(k2, k), **summ(g)) for k, g in df.groupby(k2, sort=False)]).to_csv('results/by_type.csv', index=False)
pq = pd.read_csv('../2026-11-14_product_quantization/results/summary.csv'); rows = []
for ds in ['HPatches', 'COCO_synthetic']:
    for nb in [8, 12, 16, 24, 32, 48, 64]:
        r = dict(dataset=ds, bytes=nb)
        for fam in ['scalar', 'pca']:
            for st in ['symmetric', 'asymmetric']:
                x = s[(s.dataset == ds) & (s.family == fam) & (s.setting == st) & (s.bytes == nb)].sort_values('precision_3px')
                r[f'{fam}_{st}'] = round(x.precision_3px.iloc[-1], 2) if len(x) else None
                r[f'{fam}_{st}_cfg'] = f"{int(x.dims.iloc[-1])}-D x {int(x.levels.iloc[-1])} lv {x.strategy.iloc[-1]}" if len(x) else ''
        for st, dd in [('symmetric', ['sdc_cos', 'sdc_l2']), ('asymmetric', ['adc_cos'])]:
            x = pq[(pq.dataset == ds) & (pq.bytes == nb) & pq.distance.isin(dd)].sort_values('precision_3px')
            r[f'pq_{st}'] = round(x.precision_3px.iloc[-1], 2) if len(x) else None
        rows.append(r)
pd.DataFrame(rows).to_csv('results/equal_bytes_sym_vs_asym.csv', index=False); print(pd.DataFrame(rows).to_string(index=False))
