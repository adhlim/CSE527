"""Merge PQ per-pair parts and compare, at equal bytes, with scalar quantization (2026-11-08_accuracy_matrix)
and reduce + quantize (2026-11-14_dim_reduction).
usage: python summarize.py RESULTS_DIR ACCURACY_MATRIX_SUMMARY.csv DIM_REDUCTION_EQUAL_BYTES_BEST.csv"""
import sys, glob
from pathlib import Path
import pandas as pd
OUT = Path(sys.argv[1]); AM = pd.read_csv(sys.argv[2]); DR = pd.read_csv(sys.argv[3])
df = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(OUT / 'per_pair_*_*.csv.gz')))])
assert not df.duplicated(['dataset', 'pair', 'rotation', 'M', 'K', 'distance']).any()
print('pairs', df.groupby('dataset').pair.nunique().to_dict())
def summ(g):
    r = dict(pairs=len(g), mutual=int(g.mutual.sum()), correct_3px=int(g.correct_3px.sum()))
    for px in (1, 3, 5): r[f'precision_{px}px'] = 100 * g[f'correct_{px}px'].sum() / g.mutual.sum()
    return r
keys = ['dataset', 'rotation', 'M', 'K', 'distance', 'bytes']
s = pd.DataFrame([dict(zip(keys, k), **summ(g)) for k, g in df.groupby(keys, sort=False)]); s.to_csv(OUT / 'summary.csv', index=False)
k2 = ['dataset', 'category'] + keys[1:]
pd.DataFrame([dict(zip(k2, k), **summ(g)) for k, g in df.groupby(k2, sort=False)]).to_csv(OUT / 'by_type.csv', index=False)
pd.set_option('display.width', 220)
rowsout = []
for ds in ('HPatches', 'COCO_synthetic'):
    d = s[(s.dataset == ds) & (s.K > 0)]
    print(f"\n== {ds}: float {float(s[(s.dataset == ds) & (s.K == 0)].precision_3px.iloc[0]):.2f}")
    print(d.pivot_table(index=['bytes', 'M', 'K', 'rotation'], columns='distance', values='precision_3px').round(2).to_string())
    for nb in (8, 16, 32):
        pq = d[(d.bytes == nb) & (d.distance != 'adc_cos')].sort_values('precision_3px').iloc[-1]
        adc = d[(d.bytes == nb) & (d.distance == 'adc_cos')].sort_values('precision_3px').iloc[-1]
        sc = AM[(AM.dataset == ds) & (AM.packed_bytes == nb)].sort_values('precision_3px')
        dr = DR[(DR.dataset == ds) & (DR.bytes == nb) & DR.family.str.contains('quantize')].sort_values('precision_3px')
        rowsout.append(dict(dataset=ds, bytes=nb,
            scalar64=round(sc.precision_3px.iloc[-1], 2) if len(sc) else None,
            scalar64_config=f"{int(sc.levels.iloc[-1])} levels {sc.strategy.iloc[-1]} {sc.distance.iloc[-1]}" if len(sc) else '',
            reduce_quantize=round(dr.precision_3px.iloc[-1], 2) if len(dr) else None,
            reduce_quantize_config=f"{dr.family.iloc[-1][:3]} {int(dr.dims.iloc[-1])}-D x {int(dr.bits.iloc[-1])} bit {dr.distance.iloc[-1]}" if len(dr) else '',
            pq_symmetric=round(pq.precision_3px, 2), pq_config=f"M={int(pq.M)} K={int(pq.K)} {pq.rotation} {pq.distance}",
            pq_asymmetric=round(adc.precision_3px, 2), pq_asym_config=f"M={int(adc.M)} K={int(adc.K)} {adc.rotation}"))
t = pd.DataFrame(rowsout); t.to_csv(OUT / 'equal_bytes_three_families.csv', index=False)
print('\n', t.to_string(index=False))
