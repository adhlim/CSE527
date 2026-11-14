"""Evaluate learned thresholds (best-on-dev checkpoint of each run) on the usual validation sets with the exact
hard-threshold evaluator of 2026-11-08_accuracy_matrix (code rule count(x > edge), exact MNN, smallest index on ties).
usage: python eval_learned.py CACHE_DIR ORIG_CODE_DIR RESULTS_DIR"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
CACHE, ORIG, OUT = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
sys.path.insert(0, ORIG); import features as f
CAL = np.load(CACHE / 'coco' / 'calibration.npy').astype(np.float32)


def codes(x, e):
    out = np.empty(x.shape, np.uint8)
    for d in range(64): out[:, d] = np.searchsorted(e[d], x[:, d], side='left')
    return out


def table(e, L):
    c = codes(CAL, e); t = np.zeros((64, L), np.float32)
    for d in range(64):
        n = np.bincount(c[:, d], minlength=L); s = np.bincount(c[:, d], weights=CAL[:, d], minlength=L)
        mids = np.concatenate([[e[d][0]], (e[d][:-1] + e[d][1:]) / 2, [e[d][-1]]]); t[d] = np.where(n > 0, s / np.maximum(n, 1), mids)
    return t


def mnn_from(D):
    r = D.argmin(1); c = D.argmin(0); i = np.flatnonzero(c[r] == np.arange(len(r))); return i, r[i]


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


CFG = {}
for p in sorted(glob.glob(str(OUT / 'log_*.json'))):
    d = json.load(open(p)); z = np.load(OUT / f"edges_{d['tag']}.npz"); L = d['levels']
    CFG[f"learned:{d['tag']}"] = (L, z['best'], dict(variant=d['variant'], init=d.get('init', 'quantile'), tau=d['tau'], seed=d['seed'],
                                                     best_step=d['best_step'], dev=d['best_dev'], dev_init=d['init_dev']))
    CFG.setdefault(f'quantile:L{L}', (L, np.quantile(CAL, np.arange(1, L) / L, axis=0).T.astype(np.float32), {}))
rows = []
for label, path in [('COCO_synthetic', CACHE / 'coco' / 'test_pairs.npz'), ('HPatches', CACHE / 'hpatches_cache' / 'hpatches_pairs.npz')]:
    for p in f.read_pair_cache(path):
        a, b = p['d1'].astype(np.float32), p['d2'].astype(np.float32)
        for name, (L, e, meta) in CFG.items():
            ca, cb = codes(a, e), codes(b, e); fa, fb = ca.astype(np.float32), cb.astype(np.float32); t = table(e, L) if 'tab' not in meta else meta['tab']
            meta['tab'] = t
            ta = (fa[:, :, None] > np.arange(L - 1)).reshape(len(a), -1).astype(np.float32); tb = (fb[:, :, None] > np.arange(L - 1)).reshape(len(b), -1).astype(np.float32)
            ar = np.arange(64)[None]
            for dist, Dm in [('idx_L1', ta @ (1 - tb).T + (1 - ta) @ tb.T),
                             ('idx_L2', (fa * fa).sum(1)[:, None] + (fb * fb).sum(1)[None] - 2 * fa @ fb.T),
                             ('recon_cos', -(unit(t[ar, ca]) @ unit(t[ar, cb]).T))]:
                i, j = mnn_from(Dm); err = np.linalg.norm(f.project(p['k1'][i], p['H']) - p['k2'][j], axis=1)
                rows.append(dict(dataset=label, pair=p['name'], config=name, levels=L, distance=dist, mutual=len(i), correct_3px=int((err < 3).sum())))
    print(label, 'done', flush=True)
df = pd.DataFrame(rows); df.to_csv(OUT / 'validation_per_pair.csv.gz', index=False)
s = df.groupby(['dataset', 'config', 'levels', 'distance'], as_index=False)[['mutual', 'correct_3px']].sum()
s['precision_3px'] = 100 * s.correct_3px / s.mutual
for k in ('variant', 'init', 'tau', 'seed', 'best_step', 'dev', 'dev_init'): s[k] = s.config.map(lambda c: CFG[c][2].get(k))
s.to_csv(OUT / 'validation_summary.csv', index=False)
pd.set_option('display.width', 200)
print(s.pivot_table(index=['levels', 'config'], columns=['dataset', 'distance'], values='precision_3px').round(2).to_string())
