"""2026-11-14 asymmetric matching: first image float, second image compressed (validation only).

Purpose: the PQ experiment measured an asymmetric setting (query float, database compressed) only for PQ.
This measures the same setting for 64-D scalar quantization and for PCA reduce + scalar quantization, so the
three families can be compared in that setting too.

FIXED  : regenerated 2026-10-14 XFeat cache; edges of 2026-11-08_accuracy_matrix (edges_all.npz) for the 64-D
         scalar family; PCA projection of 2026-11-14_dim_reduction; reconstruction value = calibration conditional
         mean; exact MNN, smallest index on ties; homography reprojection metric.
CHANGED: family / configuration. Distance = cosine between the float first-image descriptor and the reconstructed
         second-image descriptor (in the reduced space for PCA). A symmetric recon_cos row is recomputed for reference.

usage: python asym.py CACHE_DIR ORIG_CODE_DIR EDGES_ALL.npz PROJECTIONS.npz RESULTS_DIR coco|hpatches [start end]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

CACHE, ORIG = Path(sys.argv[1]), Path(sys.argv[2]); EDGES = np.load(sys.argv[3]); P = np.load(sys.argv[4])
OUT = Path(sys.argv[5]); WHICH = sys.argv[6]
START = int(sys.argv[7]) if len(sys.argv) > 7 else 0
END = int(sys.argv[8]) if len(sys.argv) > 8 else None
sys.path.insert(0, str(ORIG))
import features as f
CAL = np.load(CACHE / 'coco' / 'calibration.npy').astype(np.float32)
STRATS = ['uniform', 'percentile', 'quantile', 'lloyd', 'uniform_shared', 'quantile_shared']; LEVELS = [4, 8, 16, 32, 256]
DIMS = [48, 32, 24, 16]; RL = [4, 8, 16, 256]; RS = ['percentile', 'quantile']


def codes(z, e):
    out = np.empty(z.shape, np.uint8)
    for d in range(z.shape[1]): out[:, d] = np.searchsorted(e[d], z[:, d], side='left')
    return out


def table(z, e, L):
    c = codes(z, e); t = np.zeros((z.shape[1], L), np.float32)
    for d in range(z.shape[1]):
        n = np.bincount(c[:, d], minlength=L); s = np.bincount(c[:, d], weights=z[:, d], minlength=L)
        mids = np.concatenate([[e[d][0]], (e[d][:-1] + e[d][1:]) / 2, [e[d][-1]]])
        t[d] = np.where(n > 0, s / np.maximum(n, 1), mids)
    return t


def edges_for(z, L, st):
    if st == 'percentile':
        lo, hi = np.percentile(z, 0.1, axis=0), np.percentile(z, 99.9, axis=0)
        return (lo[:, None] + (hi - lo)[:, None] * (np.arange(1, L) / L)[None]).astype(np.float32)
    return np.quantile(z, np.arange(1, L) / L, axis=0).T.astype(np.float32)


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
def mnn_from(D):
    r = D.argmin(1); c = D.argmin(0); i = np.flatnonzero(c[r] == np.arange(len(r))); return i, r[i]


CFG = []                                                    # (family, dims, levels, strategy, project, edges, table)
for st in STRATS:
    for L in LEVELS:
        e = EDGES[f'{st}_{L}']; CFG.append(('scalar', 64, L, st, None, e, table(CAL, e, L)))
for m in DIMS:
    V = P['pca'][:, :m]; zc = (CAL - P['mean']) @ V
    for L in RL:
        for st in RS:
            e = edges_for(zc, L, st); CFG.append(('pca', m, L, st, V, e, table(zc, e, L)))

path = {'coco': CACHE / 'coco' / 'test_pairs.npz', 'hpatches': CACHE / 'hpatches_cache' / 'hpatches_pairs.npz'}[WHICH]
label = {'coco': 'COCO_synthetic', 'hpatches': 'HPatches'}[WHICH]
t0 = time.time(); pairs = f.read_pair_cache(path)[START:END]; rows = []
for ix, p in enumerate(pairs):
    a = np.ascontiguousarray(p['d1'], np.float32); b = np.ascontiguousarray(p['d2'], np.float32)
    scene = p['name'].split('/')[0]
    cat = 'illumination' if scene.startswith('i_') else 'viewpoint' if scene.startswith('v_') else 'synthetic'

    def record(fam, m, L, st, setting, ij):
        i, j = ij
        err = np.linalg.norm(f.project(p['k1'][i], p['H']) - p['k2'][j], axis=1)
        rows.append(dict(dataset=label, category=cat, pair=p['name'], family=fam, dims=m, levels=L, strategy=st, setting=setting,
                         bytes=(m * int(np.log2(L)) // 8 if L else 256), mutual=len(i), correct_1px=int((err < 1).sum()),
                         correct_3px=int((err < 3).sum()), correct_5px=int((err < 5).sum())))

    record('float', 64, 0, 'none', 'float', mnn_from(-(a @ b.T)))
    for fam, m, L, st, V, e, t in CFG:
        za, zb = (a, b) if V is None else ((a - P['mean']) @ V, (b - P['mean']) @ V)
        ar = np.arange(m)[None]; ra, rb = t[ar, codes(za, e)], t[ar, codes(zb, e)]
        record(fam, m, L, st, 'symmetric', mnn_from(-(unit(ra) @ unit(rb).T)))
        record(fam, m, L, st, 'asymmetric', mnn_from(-(unit(za) @ unit(rb).T)))
    if ix % 100 == 0: print(label, ix + 1, len(pairs), round(time.time() - t0), 's', flush=True)
pd.DataFrame(rows).to_csv(OUT / f'per_pair_{WHICH}_{START:04d}.csv.gz', index=False)
print('done', label, round(time.time() - t0), 's')
