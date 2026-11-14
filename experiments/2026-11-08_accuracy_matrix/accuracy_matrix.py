"""2026-11-08 accuracy matrix: calibration strategy x number of levels x distance function.

FIXED  : regenerated 2026-10-14 XFeat cache, calibration descriptors (COCO 192 images, 191,581),
         exact MNN with smallest index on ties, homography reprojection metric,
         reconstruction value of a bin = calibration conditional mean of that bin (all strategies).
CHANGED: calibration strategy, number of levels, distance function.
Validation only (COCO synthetic + HPatches). Final test (MegaDepth-1500) is NOT used here.

usage: python accuracy_matrix.py CACHE_DIR ORIG_CODE_DIR SPEED_KERNELS_DIR OUT_DIR [max_pairs]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys, json, time
from pathlib import Path
import numpy as np, pandas as pd

CACHE, ORIG, SK, OUT = map(Path, sys.argv[1:5])
MAX_PAIRS = int(sys.argv[5]) if len(sys.argv) > 5 else None
sys.path.insert(0, str(ORIG)); sys.path.insert(0, str(SK))
import features as f
import speed_kernels as sk

OUT.mkdir(parents=True, exist_ok=True)
CAL = np.load(CACHE / 'coco' / 'calibration.npy').astype(np.float32)
LEVELS = [4, 8, 16, 32, 256]
STRATEGIES = ['uniform', 'percentile', 'quantile', 'lloyd', 'uniform_shared', 'quantile_shared']
DISTANCES = ['idx_L1', 'idx_L2', 'recon_cos']
K = sk.Kernels(2)


# ---------------- calibration: per-dimension edges (64, L-1) ----------------
def lloyd_1d(v, L, iters=60):
    v = np.sort(v); cs = np.concatenate([[0.], np.cumsum(v, dtype=np.float64)])
    e = np.quantile(v, np.arange(1, L) / L)
    for _ in range(iters):
        idx = np.concatenate([[0], np.searchsorted(v, e, side='right'), [len(v)]])
        cnt = np.diff(idx); s = cs[idx[1:]] - cs[idx[:-1]]
        c = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        if np.isnan(c).any():                     # empty bin: interpolate its centre from neighbours
            c = pd.Series(c).interpolate(limit_direction='both').to_numpy()
        new = (c[:-1] + c[1:]) / 2
        if np.allclose(new, e, rtol=0, atol=1e-9): break
        e = new
    return e


def fit_edges(strategy, L):
    if strategy == 'uniform':
        lo, hi = CAL.min(0), CAL.max(0)
    elif strategy == 'percentile':
        lo, hi = np.percentile(CAL, 0.1, axis=0), np.percentile(CAL, 99.9, axis=0)
    elif strategy == 'uniform_shared':
        lo, hi = np.full(64, CAL.min()), np.full(64, CAL.max())
    if strategy in ('uniform', 'percentile', 'uniform_shared'):
        t = np.arange(1, L) / L
        return (lo[:, None] + (hi - lo)[:, None] * t[None]).astype(np.float32)
    if strategy == 'quantile':
        return np.quantile(CAL, np.arange(1, L) / L, axis=0).T.astype(np.float32)
    if strategy == 'quantile_shared':
        e = np.quantile(CAL.ravel(), np.arange(1, L) / L).astype(np.float32)
        return np.repeat(e[None], 64, 0)
    if strategy == 'lloyd':
        return np.stack([lloyd_1d(CAL[:, d].astype(np.float64), L) for d in range(64)]).astype(np.float32)
    raise ValueError(strategy)


def codes(x, edges):  # count of x > edge (same rule as the original), via searchsorted per dimension
    out = np.empty(x.shape, np.uint8)
    for d in range(64):
        out[:, d] = np.searchsorted(edges[d], x[:, d], side='left')
    return out


def recon_table(edges, L):
    c = codes(CAL, edges); t = np.zeros((64, L), np.float32); empty = 0
    for d in range(64):
        n = np.bincount(c[:, d], minlength=L); s = np.bincount(c[:, d], weights=CAL[:, d], minlength=L)
        mids = np.concatenate([[edges[d][0]], (edges[d][:-1] + edges[d][1:]) / 2, [edges[d][-1]]]) if L > 1 else np.zeros(1)
        t[d] = np.where(n > 0, s / np.maximum(n, 1), mids); empty += int((n == 0).sum())
    return t, empty


# ---------------- MNN helpers ----------------
def mnn_from(D):  # minimise; argmin returns the smallest index on ties
    r = D.argmin(1); c = D.argmin(0); i = np.flatnonzero(c[r] == np.arange(len(r))); return i, r[i]


def run_config(ca, cb, ra, rb, dist):
    if dist == 'idx_L1':
        return K.mnn('l1', ca, cb)                   # exact integer SAD kernel (verified)
    if dist == 'idx_L2':
        a = ca.astype(np.float32); b = cb.astype(np.float32)
        # integers: products <= 65025, sums <= 4.2e6 < 2^24 -> exact in float32
        D = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * (a @ b.T)
        return mnn_from(D)
    if dist == 'recon_cos':
        na = ra / np.maximum(np.linalg.norm(ra, axis=1, keepdims=True), 1e-12)
        nb = rb / np.maximum(np.linalg.norm(rb, axis=1, keepdims=True), 1e-12)
        return mnn_from(-(na @ nb.T))
    raise ValueError(dist)


# ---------------- fit everything once ----------------
t0 = time.time(); configs = {}; fitinfo = []
for s in STRATEGIES:
    for L in LEVELS:
        e = fit_edges(s, L); tab, empty = recon_table(e, L)
        configs[(s, L)] = (e, tab)
        cal_codes = codes(CAL, e)
        occ = np.array([np.bincount(cal_codes[:, d], minlength=L) for d in range(64)]) / len(CAL)
        ent = float(np.mean([-(p[p > 0] * np.log2(p[p > 0])).sum() for p in occ]))
        mse = float(np.mean((tab[np.arange(64)[None], cal_codes] - CAL) ** 2))
        fitinfo.append(dict(strategy=s, levels=L, bits=int(np.log2(L)), packed_bytes=int(64 * np.log2(L) / 8),
                            empty_bins=empty, mean_code_entropy_bits=ent, calibration_recon_mse=mse))
np.savez(OUT / 'edges_all.npz', **{f'{s}_{L}': configs[(s, L)][0] for (s, L) in configs})
pd.DataFrame(fitinfo).to_csv(OUT / 'calibration_fit.csv', index=False)
print('fitted', len(configs), 'configs in', round(time.time() - t0, 1), 's', flush=True)

# sanity: the quantile-8 edges equal the saved 2026-10-14 edges up to float noise
if (SK / 'inputs' / 'fixed_quantile_edges.npz').exists():
    saved = np.load(SK / 'inputs' / 'fixed_quantile_edges.npz')['bins_8']
    print('quantile-8 vs saved edges max abs diff', float(np.abs(saved - configs[('quantile', 8)][0]).max()), flush=True)

rows = []
for label, path in [('COCO_synthetic', CACHE / 'coco' / 'test_pairs.npz'),
                    ('HPatches', CACHE / 'hpatches_cache' / 'hpatches_pairs.npz')]:
    pairs = f.read_pair_cache(path)[:MAX_PAIRS]
    for ix, p in enumerate(pairs):
        a = np.ascontiguousarray(p['d1'], np.float32); b = np.ascontiguousarray(p['d2'], np.float32)
        scene = p['name'].split('/')[0]
        cat = 'illumination' if scene.startswith('i_') else 'viewpoint' if scene.startswith('v_') else 'synthetic'

        def record(strategy, L, dist, ij):
            i, j = ij
            err = np.linalg.norm(f.project(p['k1'][i], p['H']) - p['k2'][j], axis=1)
            rows.append(dict(dataset=label, category=cat, pair=p['name'], strategy=strategy, levels=L, distance=dist,
                             mutual=len(i), correct_1px=int((err < 1).sum()), correct_3px=int((err < 3).sum()),
                             correct_5px=int((err < 5).sum())))

        record('float', 0, 'cosine', mnn_from(-(a @ b.T)))
        for (s, L), (e, tab) in configs.items():
            ca, cb = codes(a, e), codes(b, e)
            ra = tab[np.arange(64)[None], ca]; rb = tab[np.arange(64)[None], cb]
            for dist in DISTANCES:
                ij = run_config(ca, cb, ra, rb, dist)
                if ix == 0 and dist != 'recon_cos':   # exact check against NumPy integer reference
                    ref = sk.ref_mnn_min(sk.ref_dist({'idx_L1': 'l1', 'idx_L2': 'l2'}[dist], ca, cb))
                    assert all(np.array_equal(x, y) for x, y in zip(ij, ref)), (s, L, dist)
                record(s, L, dist, ij)
        if ix % 50 == 0: print(label, ix + 1, len(pairs), round(time.time() - t0), 's', flush=True)

df = pd.DataFrame(rows); df.to_csv(OUT / 'per_pair.csv.gz', index=False)
keys = ['dataset', 'strategy', 'levels', 'distance']


def summ(g):
    r = dict(pairs=len(g), mutual=int(g.mutual.sum()), correct_3px=int(g.correct_3px.sum()))
    for px in (1, 3, 5): r[f'precision_{px}px'] = 100 * g[f'correct_{px}px'].sum() / g.mutual.sum()
    return r


summary = pd.DataFrame([dict(zip(keys, k), **summ(g)) for k, g in df.groupby(keys, sort=False)])
summary = summary.merge(pd.DataFrame(fitinfo)[['strategy', 'levels', 'packed_bytes']], on=['strategy', 'levels'], how='left')
summary.loc[summary.strategy == 'float', 'packed_bytes'] = 256
summary.to_csv(OUT / 'summary.csv', index=False)
k2 = ['dataset', 'category', 'strategy', 'levels', 'distance']
pd.DataFrame([dict(zip(k2, k), **summ(g)) for k, g in df.groupby(k2, sort=False)]).to_csv(OUT / 'by_type.csv', index=False)
(OUT / 'environment.json').write_text(json.dumps(dict(
    date='2026-11-08', strategies=STRATEGIES, levels=LEVELS, distances=DISTANCES, percentile=[0.1, 99.9],
    lloyd='per-dimension 1D Lloyd-Max, init = quantile edges, <=60 iterations',
    reconstruction='calibration conditional mean per bin (bin midpoint if empty)',
    code_rule='count of x > edge', tie_rule='smallest index', max_pairs=MAX_PAIRS,
    data_status='validation only (COCO synthetic, HPatches); MegaDepth-1500 reserved as final test'), indent=2))
print('total minutes', round((time.time() - t0) / 60, 1))
