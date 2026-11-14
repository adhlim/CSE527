"""2026-11-14 SIFT generalization: the XFeat accuracy matrix and the PCA + quantize comparison repeated on SIFT (128-D).

FIXED  : SIFT cache of extract_sift.py (same images / homographies as the XFeat validation sets); calibration = SIFT
         descriptors of the 192 COCO calibration images; code rule count(x > edge); exact MNN, smallest index on ties;
         pooled precision at 3 px. Descriptor = raw SIFT, L2-normalised (DESC=sift) or RootSIFT (DESC=rootsift).
CHANGED: same axes as 2026-11-08_accuracy_matrix: calibration strategy (6), levels (4, 8, 16, 256), distance
         (idx_L1 only for 4 and 8 levels, idx_L2, recon_cos); plus PCA to 64 / 32 dimensions with quantile edges,
         and the native uint8 SIFT descriptor with L2 (how SIFT is normally stored, 128 B).
Validation only.

usage: python sift_matrix.py SIFT_CACHE_DIR ORIG_CODE_DIR OUT_DIR DESC fit
       python sift_matrix.py SIFT_CACHE_DIR ORIG_CODE_DIR OUT_DIR DESC coco|hpatches [start end]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys, time
from pathlib import Path
import numpy as np, pandas as pd
SC, ORIG, OUT, DESC, MODE = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3]), sys.argv[4], sys.argv[5]
START = int(sys.argv[6]) if len(sys.argv) > 6 else 0
END = int(sys.argv[7]) if len(sys.argv) > 7 else None
sys.path.insert(0, ORIG); import features as f
OUT.mkdir(parents=True, exist_ok=True)
D = 128; LEVELS = [4, 8, 16, 256]; STRATS = ['uniform', 'percentile', 'quantile', 'lloyd', 'uniform_shared', 'quantile_shared']
PCA_M = [64, 32]; PCA_L = [4, 16, 256]


def prep(u8):
    x = u8.astype(np.float32)
    if DESC == 'rootsift': x = np.sqrt(x / np.maximum(x.sum(1, keepdims=True), 1e-12))
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def lloyd_1d(v, L, iters=60):
    v = np.sort(v); cs = np.concatenate([[0.], np.cumsum(v, dtype=np.float64)]); e = np.quantile(v, np.arange(1, L) / L)
    for _ in range(iters):
        idx = np.concatenate([[0], np.searchsorted(v, e, side='right'), [len(v)]]); cnt = np.diff(idx); s = cs[idx[1:]] - cs[idx[:-1]]
        c = np.where(cnt > 0, s / np.maximum(cnt, 1), np.nan)
        if np.isnan(c).any(): c = pd.Series(c).interpolate(limit_direction='both').to_numpy()
        new = (c[:-1] + c[1:]) / 2
        if np.allclose(new, e, rtol=0, atol=1e-9): break
        e = new
    return e


def fit_edges(z, st, L):
    n = z.shape[1]; t = np.arange(1, L) / L
    if st == 'uniform': lo, hi = z.min(0), z.max(0)
    elif st == 'percentile': lo, hi = np.percentile(z, 0.1, axis=0), np.percentile(z, 99.9, axis=0)
    elif st == 'uniform_shared': lo, hi = np.full(n, z.min()), np.full(n, z.max())
    if st in ('uniform', 'percentile', 'uniform_shared'): return (lo[:, None] + (hi - lo)[:, None] * t[None]).astype(np.float32)
    if st == 'quantile': return np.quantile(z, t, axis=0).T.astype(np.float32)
    if st == 'quantile_shared': return np.repeat(np.quantile(z.ravel(), t).astype(np.float32)[None], n, 0)
    return np.stack([lloyd_1d(z[:, d].astype(np.float64), L) for d in range(n)]).astype(np.float32)


def codes(z, e):
    if e.shape[1] <= 15: return (z[:, :, None] > e[None]).sum(-1).astype(np.int64)
    out = np.empty(z.shape, np.int64)
    for d in range(z.shape[1]): out[:, d] = np.searchsorted(e[d], z[:, d], side='left')
    return out


def table(z, e, L):
    c = codes(z, e); t = np.zeros((z.shape[1], L), np.float32); empty = 0
    for d in range(z.shape[1]):
        n = np.bincount(c[:, d], minlength=L); s = np.bincount(c[:, d], weights=z[:, d], minlength=L)
        mids = np.concatenate([[e[d][0]], (e[d][:-1] + e[d][1:]) / 2, [e[d][-1]]]); t[d] = np.where(n > 0, s / np.maximum(n, 1), mids); empty += int((n == 0).sum())
    return t, empty


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
def mnn_from(Dm):
    r = Dm.argmin(1); c = Dm.argmin(0); i = np.flatnonzero(c[r] == np.arange(len(r))); return i, r[i]
def l2(a, b): return (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T


FIT = OUT / f'fit_{DESC}.npz'
if MODE == 'fit':
    cal = prep(np.load(SC / 'calibration_u8.npy')); t0 = time.time(); save = {}; info = []
    for st in STRATS:
        for L in LEVELS:
            e = fit_edges(cal, st, L); t, empty = table(cal, e, L); save[f's_{st}_{L}_e'] = e; save[f's_{st}_{L}_t'] = t
            c = codes(cal, e); ent = float(np.mean([-(p[p > 0] * np.log2(p[p > 0])).sum() for p in (np.array([np.bincount(c[:, d], minlength=L) for d in range(D)]) / len(cal))]))
            info.append(dict(family='scalar', dims=D, strategy=st, levels=L, empty_bins=empty, mean_code_entropy_bits=ent))
        print(st, round(time.time() - t0), 's', flush=True)
    mean = cal.mean(0); w, V = np.linalg.eigh(np.cov((cal - mean).T)); o = np.argsort(w)[::-1]; V = V[:, o].astype(np.float32); w = w[o]
    save['mean'] = mean; save['V'] = V
    for m in PCA_M:
        z = (cal - mean) @ V[:, :m]
        for L in PCA_L:
            e = fit_edges(z, 'quantile', L); t, empty = table(z, e, L); save[f'p_{m}_{L}_e'] = e; save[f'p_{m}_{L}_t'] = t
            info.append(dict(family='pca', dims=m, strategy='quantile', levels=L, empty_bins=empty, variance_kept=float(w[:m].sum() / w.sum())))
    np.savez(FIT, **save); pd.DataFrame(info).to_csv(OUT / f'fit_info_{DESC}.csv', index=False); print('fit done', round(time.time() - t0), 's'); sys.exit()

F = np.load(FIT); mean, V = F['mean'], F['V']
label = {'coco': 'COCO_synthetic', 'hpatches': 'HPatches'}[MODE]
z_ = np.load(SC / f'{MODE}_pairs.npz'); n_all = int(z_['count']); idxs = range(n_all)[START:END]; rows = []; t0 = time.time()
for ix, pi in enumerate(idxs):
    p = {k: z_[f'{pi}_{k}'] for k in ('d1', 'd2', 'k1', 'k2', 'H')}; name = str(z_[f'{pi}_name'])
    scene = name.split('/')[0]; cat = 'illumination' if scene.startswith('i_') else 'viewpoint' if scene.startswith('v_') else 'synthetic'
    a, b = prep(p['d1']), prep(p['d2'])

    def record(fam, m, st, L, dist, nbytes, ij):
        i, j = ij; err = np.linalg.norm(f.project(p['k1'][i], p['H']) - p['k2'][j], axis=1)
        rows.append(dict(dataset=label, category=cat, pair=name, descriptor=DESC, family=fam, dims=m, strategy=st, levels=L, distance=dist,
                         bytes=nbytes, mutual=len(i), correct_1px=int((err < 1).sum()), correct_3px=int((err < 3).sum()), correct_5px=int((err < 5).sum())))

    record('float', D, 'none', 0, 'cosine', 4 * D, mnn_from(-(a @ b.T)))
    if DESC == 'sift': record('native_u8', D, 'none', 256, 'l2', D, mnn_from(l2(p['d1'].astype(np.float32), p['d2'].astype(np.float32))))
    for st in STRATS:
        for L in LEVELS:
            e, t = F[f's_{st}_{L}_e'], F[f's_{st}_{L}_t']; ca, cb = codes(a, e), codes(b, e); nb = D * int(np.log2(L)) // 8; ar = np.arange(D)[None]
            fa, fb = ca.astype(np.float32), cb.astype(np.float32)
            record('scalar', D, st, L, 'idx_L2', nb, mnn_from(l2(fa, fb)))
            record('scalar', D, st, L, 'recon_cos', nb, mnn_from(-(unit(t[ar, ca]) @ unit(t[ar, cb]).T)))
            if L <= 8:
                ta = (ca[:, :, None] > np.arange(L - 1)).reshape(len(a), -1).astype(np.float32); tb = (cb[:, :, None] > np.arange(L - 1)).reshape(len(b), -1).astype(np.float32)
                record('scalar', D, st, L, 'idx_L1', nb, mnn_from(ta @ (1 - tb).T + (1 - ta) @ tb.T))
    for m in PCA_M:
        za, zb = (a - mean) @ V[:, :m], (b - mean) @ V[:, :m]
        record('pca_float', m, 'none', 0, 'cosine', 4 * m, mnn_from(-(unit(za) @ unit(zb).T)))
        for L in PCA_L:
            e, t = F[f'p_{m}_{L}_e'], F[f'p_{m}_{L}_t']; ca, cb = codes(za, e), codes(zb, e); ar = np.arange(m)[None]; nb = m * int(np.log2(L)) // 8
            record('pca', m, 'quantile', L, 'idx_L2', nb, mnn_from(l2(ca.astype(np.float32), cb.astype(np.float32))))
            record('pca', m, 'quantile', L, 'recon_cos', nb, mnn_from(-(unit(t[ar, ca]) @ unit(t[ar, cb]).T)))
    if ix % 50 == 0: print(label, ix + 1, len(idxs), round(time.time() - t0), 's', flush=True)
pd.DataFrame(rows).to_csv(OUT / f'per_pair_{DESC}_{MODE}_{START:04d}.csv.gz', index=False); print('done', label, START, round(time.time() - t0), 's')
