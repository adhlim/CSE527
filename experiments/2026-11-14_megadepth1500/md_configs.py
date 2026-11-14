"""The 15 pre-registered configurations (C0..C14) as pure NumPy functions on a frozen parameter file.

`load(path)` reads frozen_configs.npz; `match(cid, P, a, b)` returns the mutual-nearest-neighbour index pairs (i, j)
for L2-normalised float descriptors a (n1, 64) and b (n2, 64). Ties go to the smallest index (NumPy argmin).
Code rule: code = count(x > edge). Nothing here is fitted at run time.
"""
import numpy as np

CONFIGS = {
    'C0': ('float32, 64-D, cosine', 256),
    'C1': ('int8 per-tensor, integer dot', 64),
    'C2': ('64-D, 16 levels, percentile, index-L2', 32),
    'C3': ('64-D, 8 levels, quantile, index-L1 (thermometer)', 24),
    'C4': ('64-D, 8 levels, quantile, index-L2', 24),
    'C5': ('64-D, 8 levels, quantile, reconstruction cosine', 24),
    'C6': ('64-D, 4 levels, quantile, index-L2', 16),
    'C7': ('64-D, 4 levels, uniform, index-L2', 16),
    'C8': ('64-D, 4 levels, learned thresholds, index-L2', 16),
    'C9': ('PCA 32-D, 16 levels, quantile, reconstruction cosine', 16),
    'C10': ('GFR 32-D, 16 levels, quantile, reconstruction cosine', 16),
    'C11': ('PCA 48-D, 16 levels, quantile, reconstruction cosine', 24),
    'C12': ('PQ, PCA rotation, M=16, K=256, reconstructed L2', 16),
    'C13': ('learned projection 32-D + learned thresholds, 16 levels, index-L2', 16),
    'C14': ('PCA 32-D, float, cosine', 128),
}


def load(path):
    with np.load(path) as z: return {k: z[k] for k in z.files}


def codes(z, e):
    out = np.empty(z.shape, np.int64)
    for d in range(z.shape[1]): out[:, d] = np.searchsorted(e[d], z[:, d], side='left')
    return out


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def mnn(D):
    r = D.argmin(1); c = D.argmin(0); i = np.flatnonzero(c[r] == np.arange(len(r))); return i, r[i]


def l2(a, b): return (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T


def _idx(a, b, e, kind):
    fa, fb = codes(a, e).astype(np.float32), codes(b, e).astype(np.float32)
    if kind == 'l2': return mnn(l2(fa, fb))                      # integers: exact in float32 (sums < 2^24)
    L = e.shape[1] + 1
    ta = (fa[:, :, None] > np.arange(L - 1, dtype=np.float32)).reshape(len(a), -1).astype(np.float32)
    tb = (fb[:, :, None] > np.arange(L - 1, dtype=np.float32)).reshape(len(b), -1).astype(np.float32)
    return mnn(ta @ (1 - tb).T + (1 - ta) @ tb.T)                # Hamming distance of thermometer bits = index-L1


def _recon(za, zb, e, t):
    ar = np.arange(za.shape[1])[None]
    return mnn(-(unit(t[ar, codes(za, e)]) @ unit(t[ar, codes(zb, e)]).T))


def _pq(z, books):
    M = books.shape[0]; ds = z.shape[1] // M; out = np.empty_like(z)
    for s in range(M):
        x = z[:, s * ds:(s + 1) * ds]; c = books[s]
        out[:, s * ds:(s + 1) * ds] = c[((x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None]).argmin(1)]
    return out


def match(cid, P, a, b):
    a = np.ascontiguousarray(a, np.float32); b = np.ascontiguousarray(b, np.float32)
    if cid == 'C0': return mnn(-(a @ b.T))
    if cid == 'C1':
        s = float(P['int8_scale']); qa = np.clip(np.rint(a / s), -127, 127).astype(np.float32); qb = np.clip(np.rint(b / s), -127, 127).astype(np.float32)
        return mnn(-(qa @ qb.T))                                  # exact: |dot| <= 64 * 127^2 < 2^24
    if cid == 'C2': return _idx(a, b, P['e_percentile_16'], 'l2')
    if cid == 'C3': return _idx(a, b, P['e_quantile_8'], 'l1')
    if cid == 'C4': return _idx(a, b, P['e_quantile_8'], 'l2')
    if cid == 'C5': return _recon(a, b, P['e_quantile_8'], P['t_quantile_8'])
    if cid == 'C6': return _idx(a, b, P['e_quantile_4'], 'l2')
    if cid == 'C7': return _idx(a, b, P['e_uniform_4'], 'l2')
    if cid == 'C8': return _idx(a, b, P['e_learned_4'], 'l2')
    if cid in ('C9', 'C10', 'C11'):
        k = {'C9': 'pca32', 'C10': 'gfr32', 'C11': 'pca48'}[cid]; W = P[f'W_{k}']
        return _recon((a - P['mean']) @ W, (b - P['mean']) @ W, P[f'e_{k}_16'], P[f't_{k}_16'])
    if cid == 'C12':
        V, pm, inv = P['pq_V'], P['pq_perm'], P['pq_inv']
        xa = _pq(((a - P['pq_mean']) @ V)[:, pm], P['pq_books'])[:, inv] @ V.T + P['pq_mean']
        xb = _pq(((b - P['pq_mean']) @ V)[:, pm], P['pq_books'])[:, inv] @ V.T + P['pq_mean']
        return mnn(l2(xa, xb))
    if cid == 'C13':
        W = P['lp_W']; return _idx((a - P['lp_mean']) @ W, (b - P['lp_mean']) @ W, P['lp_edges'], 'l2')
    if cid == 'C14':
        W = P['W_pca32']; return mnn(-(unit((a - P['mean']) @ W) @ unit((b - P['mean']) @ W).T))
    raise KeyError(cid)
