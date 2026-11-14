"""Evaluate learned projection + thresholds (best-on-dev checkpoint) and their PCA + quantile starting points on the
validation sets with exact MNN (smallest index on ties). Distances: idx_L2 (bin index, the training distance) and
recon_cos (cosine of calibration conditional means in the projected space).
usage: python eval_proj.py CACHE_DIR ORIG_CODE_DIR RESULTS_DIR"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
CACHE, ORIG, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, ORIG); import features as f
CAL = np.load(CACHE / 'coco' / 'calibration.npy').astype(np.float32)


def codes(z, e):
    out = np.empty(z.shape, np.int64)
    for d in range(z.shape[1]): out[:, d] = np.searchsorted(e[d], z[:, d], side='left')
    return out


def table(z, e, L):
    c = codes(z, e); t = np.zeros((z.shape[1], L), np.float32)
    for d in range(z.shape[1]):
        n = np.bincount(c[:, d], minlength=L); s = np.bincount(c[:, d], weights=z[:, d], minlength=L)
        mids = np.concatenate([[e[d][0]], (e[d][:-1] + e[d][1:]) / 2, [e[d][-1]]]); t[d] = np.where(n > 0, s / np.maximum(n, 1), mids)
    return t


def mnn_from(D):
    r = D.argmin(1); c = D.argmin(0); i = np.flatnonzero(c[r] == np.arange(len(r))); return i, r[i]


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


CFG = []
for p in sorted(glob.glob(str(OUT / 'log_*.json'))):
    d = json.load(open(p)); z = np.load(OUT / f"model_{d['tag']}.npz"); mean = z['mean']
    for which in ('best', 'init'):
        if which == 'init' and d['train'] != 'both' and any(c[0] == f"init:m{d['m']}_L{d['levels']}" for c in CFG): continue
        W, e = z[f'W_{which}'], z[f'edges_{which}']
        name = f"learned:{d['tag']}" if which == 'best' else f"init:m{d['m']}_L{d['levels']}"
        if any(c[0] == name for c in CFG): continue
        CFG.append((name, d['m'], d['levels'], d['bytes'], d['train'] if which == 'best' else 'none', mean, W, e, table((CAL - mean) @ W, e, d['levels'])))
rows = []
for label, path in [('COCO_synthetic', CACHE / 'coco' / 'test_pairs.npz'), ('HPatches', CACHE / 'hpatches_cache' / 'hpatches_pairs.npz')]:
    for p in f.read_pair_cache(path):
        a, b = p['d1'].astype(np.float32), p['d2'].astype(np.float32)
        scene = p['name'].split('/')[0]; cat = 'illumination' if scene.startswith('i_') else 'viewpoint' if scene.startswith('v_') else 'synthetic'
        for name, m, L, nb, trn, mean, W, e, t in CFG:
            ca, cb = codes((a - mean) @ W, e), codes((b - mean) @ W, e); fa, fb = ca.astype(np.float32), cb.astype(np.float32); ar = np.arange(m)[None]
            for dist, Dm in [('idx_L2', (fa * fa).sum(1)[:, None] + (fb * fb).sum(1)[None] - 2 * fa @ fb.T),
                             ('recon_cos', -(unit(t[ar, ca]) @ unit(t[ar, cb]).T))]:
                i, j = mnn_from(Dm); err = np.linalg.norm(f.project(p['k1'][i], p['H']) - p['k2'][j], axis=1)
                rows.append(dict(dataset=label, category=cat, pair=p['name'], config=name, dims=m, levels=L, bytes=nb, trained=trn, distance=dist,
                                 mutual=len(i), correct_3px=int((err < 3).sum())))
    print(label, 'done', flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / 'validation_per_pair.csv.gz', index=False)
for keys, fn in [(['dataset', 'config', 'dims', 'levels', 'bytes', 'trained', 'distance'], 'validation_summary.csv'),
                 (['dataset', 'category', 'config', 'dims', 'levels', 'bytes', 'trained', 'distance'], 'validation_by_type.csv')]:
    s = df.groupby(keys, as_index=False)[['mutual', 'correct_3px']].sum(); s['precision_3px'] = 100 * s.correct_3px / s.mutual; s.to_csv(OUT / fn, index=False)
s = pd.read_csv(OUT / 'validation_summary.csv'); pd.set_option('display.width', 200)
print(s.pivot_table(index=['bytes', 'dims', 'levels', 'config'], columns=['dataset', 'distance'], values='precision_3px').round(2).to_string())
