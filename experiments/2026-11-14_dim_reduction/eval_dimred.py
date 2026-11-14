"""2026-11-14 dimensionality reduction vs scalar quantization at equal bytes (validation only).

FIXED  : regenerated 2026-10-14 XFeat cache, calibration descriptors (COCO 192 images), exact MNN with the
         smallest index on ties, homography reprojection metric (same as 2026-11-08_accuracy_matrix).
CHANGED: projection (PCA / GFR / truncate = first m raw dimensions), kept dimensions m, and optional
         scalar quantization of the projected coordinates (per dimension; percentile 0.1-99.9 % or quantile).
Reduced float descriptors are L2-normalised and matched by cosine (= L2 on unit vectors).

usage: python eval_dimred.py CACHE_DIR ORIG_CODE_DIR RESULTS_DIR DATASET(coco|hpatches) [start end]
(start/end slice the pair list so that one call stays short; summarize.py merges the parts)
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys, time
from pathlib import Path
import numpy as np, pandas as pd

CACHE, ORIG, OUT = map(Path, sys.argv[1:4]); WHICH = sys.argv[4]
START = int(sys.argv[5]) if len(sys.argv) > 5 else 0
END = int(sys.argv[6]) if len(sys.argv) > 6 else None
sys.path.insert(0, str(ORIG))
import features as f

P = np.load(OUT / 'projections.npz'); MEAN = P['mean']
CAL = np.load(CACHE / 'coco' / 'calibration.npy').astype(np.float32)
DIMS = [48, 32, 24, 16]; LEVELS = [4, 8, 16, 256]; STRATS = ['percentile', 'quantile']
PROJ = {'pca': P['pca'], 'gfr': P['gfr'], 'truncate': np.eye(64, dtype=np.float32)}


def reduce(x, name, m):
    if name == 'truncate': return x[:, :m]
    return (x - MEAN) @ PROJ[name][:, :m]


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def mnn_from(D):
    r = D.argmin(1); c = D.argmin(0); i = np.flatnonzero(c[r] == np.arange(len(r))); return i, r[i]


def fit_q(z, L, strat):                # edges + conditional-mean reconstruction table, per dimension
    if strat == 'percentile':
        lo, hi = np.percentile(z, 0.1, axis=0), np.percentile(z, 99.9, axis=0)
        e = (lo[:, None] + (hi - lo)[:, None] * (np.arange(1, L) / L)[None]).astype(np.float32)
    else:
        e = np.quantile(z, np.arange(1, L) / L, axis=0).T.astype(np.float32)
    c = codes(z, e); t = np.zeros((z.shape[1], L), np.float32)
    for d in range(z.shape[1]):
        n = np.bincount(c[:, d], minlength=L); s = np.bincount(c[:, d], weights=z[:, d], minlength=L)
        mids = np.concatenate([[e[d][0]], (e[d][:-1] + e[d][1:]) / 2, [e[d][-1]]])
        t[d] = np.where(n > 0, s / np.maximum(n, 1), mids)
    return e, t


def codes(z, e):                       # count of z > edge (same rule as before)
    out = np.empty(z.shape, np.uint8)
    for d in range(z.shape[1]): out[:, d] = np.searchsorted(e[d], z[:, d], side='left')
    return out


Q = {}
for name in ('pca', 'gfr'):
    for m in DIMS:
        zc = reduce(CAL, name, m).astype(np.float32)
        for L in LEVELS:
            for st in STRATS: Q[(name, m, L, st)] = fit_q(zc, L, st)

path = {'coco': CACHE / 'coco' / 'test_pairs.npz', 'hpatches': CACHE / 'hpatches_cache' / 'hpatches_pairs.npz'}[WHICH]
label = {'coco': 'COCO_synthetic', 'hpatches': 'HPatches'}[WHICH]
t0 = time.time(); pairs = f.read_pair_cache(path)[START:END]; rows = []
for ix, p in enumerate(pairs):
    a = np.ascontiguousarray(p['d1'], np.float32); b = np.ascontiguousarray(p['d2'], np.float32)
    scene = p['name'].split('/')[0]
    cat = 'illumination' if scene.startswith('i_') else 'viewpoint' if scene.startswith('v_') else 'synthetic'

    def record(method, m, levels, dist, nbytes, ij, strat='none'):
        i, j = ij
        err = np.linalg.norm(f.project(p['k1'][i], p['H']) - p['k2'][j], axis=1)
        rows.append(dict(dataset=label, category=cat, pair=p['name'], method=method, dims=m, levels=levels, strategy=strat, distance=dist,
                         bytes=nbytes, mutual=len(i), correct_1px=int((err < 1).sum()), correct_3px=int((err < 3).sum()),
                         correct_5px=int((err < 5).sum())))

    record('float', 64, 0, 'cosine', 256, mnn_from(-(a @ b.T)))
    if ix == 0:                        # sanity: a full 64-D rotation must not change the matches
        za, zb = (a - MEAN) @ P['pca'], (b - MEAN) @ P['pca']
        da = ((a[:, None] - b[None]) ** 2).sum(-1); dz = ((za[:, None] - zb[None]) ** 2).sum(-1)
        assert np.abs(da - dz).max() < 1e-4, 'PCA-64 is not an isometry'
    for name in PROJ:
        for m in DIMS:
            za, zb = reduce(a, name, m), reduce(b, name, m)
            record(name, m, 0, 'cosine', 4 * m, mnn_from(-(unit(za) @ unit(zb).T)))
            if name == 'truncate': continue
            for L in LEVELS:
                for st in STRATS:
                    e, t = Q[(name, m, L, st)]; ca, cb = codes(za, e), codes(zb, e)
                    nb = m * int(np.log2(L)) // 8
                    fa, fb = ca.astype(np.float32), cb.astype(np.float32)      # exact: sums < 2^24
                    record(name, m, L, 'idx_L2', nb, mnn_from((fa * fa).sum(1)[:, None] + (fb * fb).sum(1)[None] - 2 * fa @ fb.T), st)
                    ra, rb = t[np.arange(m)[None], ca], t[np.arange(m)[None], cb]
                    record(name, m, L, 'recon_cos', nb, mnn_from(-(unit(ra) @ unit(rb).T)), st)
    if ix % 100 == 0: print(label, ix + 1, len(pairs), round(time.time() - t0), 's', flush=True)
pd.DataFrame(rows).to_csv(OUT / f'per_pair_{WHICH}_{START:04d}.csv.gz', index=False)
print('done', label, round(time.time() - t0), 's')
