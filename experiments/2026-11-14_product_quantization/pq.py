"""2026-11-14 product quantization (PQ) baseline: k-means PQ without any learning beyond k-means (validation only).

FIXED  : regenerated 2026-10-14 XFeat cache, codebooks fitted on COCO calibration descriptors only (k-means on a fixed
         random subset of 65,536 of the 191,581, i.e. 256 points per centroid at K = 256, the FAISS default),
         exact MNN with the smallest index on ties, homography reprojection metric (same as 2026-11-08_accuracy_matrix).
CHANGED: number of sub-vectors M (consecutive dimensions, 64/M each), codebook size K, optional PCA rotation before
         splitting (pca64 = full rotation, variance-sorted then interleaved so sub-vectors get similar variance),
         and the distance: sdc_cos (both sides reconstructed, cosine), sdc_l2 (both reconstructed, squared L2),
         adc_cos (first image float, second image reconstructed; upper bound when only the database is compressed).
Bytes = M * log2(K) / 8.

usage: python pq.py CACHE_DIR ORIG_CODE_DIR RESULTS_DIR fit none|pca64      (one rotation per call)
       python pq.py CACHE_DIR ORIG_CODE_DIR RESULTS_DIR coco|hpatches [start end]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2'); os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd

CACHE, ORIG, OUT = map(Path, sys.argv[1:4]); MODE = sys.argv[4]
START = int(sys.argv[5]) if len(sys.argv) > 5 and MODE != 'fit' else 0
END = int(sys.argv[6]) if len(sys.argv) > 6 and MODE != 'fit' else None
sys.path.insert(0, str(ORIG))
import features as f

CONFIGS = [(8, 256), (16, 256), (32, 256), (16, 16), (32, 16), (64, 16), (64, 4)]   # (M, K)
ROT = ['none', 'pca64']
SEED = 20261114; ITERS = 25; NTRAIN = 65536


def kmeans(x, K, rng):
    c = x[rng.choice(len(x), K, replace=False)].copy()
    for _ in range(ITERS):
        d = (x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None]
        a = d.argmin(1); n = np.bincount(a, minlength=K)
        s = np.stack([np.bincount(a, weights=x[:, j], minlength=K) for j in range(x.shape[1])], 1)
        empty = n == 0
        c = np.where(empty[:, None], x[rng.choice(len(x), K, replace=False)], s / np.maximum(n, 1)[:, None])
    d = (x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None]
    return c.astype(np.float32), float(d.min(1).mean())


def rotation(name, cal):
    if name == 'none': return np.zeros(64, np.float32), np.eye(64, dtype=np.float32)
    mean = cal.mean(0); w, V = np.linalg.eigh(np.cov((cal - mean).T)); V = V[:, np.argsort(w)[::-1]]
    return mean.astype(np.float32), V.astype(np.float32)


def perm_for(name, M):                 # which rotated dimensions go to which sub-vector
    if name == 'none': return np.arange(64)
    return np.arange(64).reshape(64 // M, M).T.ravel()   # interleave: sub-vector s gets dims s, s+M, s+2M, ...


def encode(z, books):
    M = len(books); ds = z.shape[1] // M; out = np.empty((len(z), M), np.int64)
    for s, c in enumerate(books):
        x = z[:, s * ds:(s + 1) * ds]
        out[:, s] = ((x * x).sum(1)[:, None] - 2 * x @ c.T + (c * c).sum(1)[None]).argmin(1)
    return out


def decode(code, books):
    return np.concatenate([c[code[:, s]] for s, c in enumerate(books)], 1)


def unit(x): return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
def mnn_from(D):
    r = D.argmin(1); c = D.argmin(0); i = np.flatnonzero(c[r] == np.arange(len(r))); return i, r[i]


if MODE == 'fit':
    OUT.mkdir(parents=True, exist_ok=True)
    cal = np.load(CACHE / 'coco' / 'calibration.npy').astype(np.float32); t0 = time.time(); save = {}; info = []
    sub = np.random.default_rng(SEED).choice(len(cal), NTRAIN, replace=False)
    for rot in [sys.argv[5]]:
        mean, V = rotation(rot, cal); save[f'{rot}_mean'] = mean; save[f'{rot}_V'] = V
        zc = ((cal - mean) @ V)[sub]
        for M, K in CONFIGS:
            z = zc[:, perm_for(rot, M)]; ds = 64 // M; rng = np.random.default_rng(SEED); mse = 0.0
            for s in range(M):
                c, e = kmeans(z[:, s * ds:(s + 1) * ds], K, rng); save[f'{rot}_{M}_{K}_{s}'] = c; mse += e
            info.append(dict(rotation=rot, M=M, K=K, bytes=M * int(np.log2(K)) // 8, calibration_recon_mse_per_dim=mse / 64))
            print(info[-1], round(time.time() - t0), 's', flush=True)
    np.savez(OUT / f'codebooks_{sys.argv[5]}.npz', **save); pd.DataFrame(info).to_csv(OUT / f'fit_info_{sys.argv[5]}.csv', index=False)
    (OUT / 'environment.json').write_text(json.dumps(dict(date='2026-11-14', configs=CONFIGS, rotations=ROT, seed=SEED,
        kmeans=f'Lloyd, random init, {ITERS} iterations, empty clusters re-seeded, {NTRAIN} training points', numpy=np.__version__), indent=2))
    sys.exit()

B = {}
for rot in ROT: B.update(np.load(OUT / f'codebooks_{rot}.npz'))
path = {'coco': CACHE / 'coco' / 'test_pairs.npz', 'hpatches': CACHE / 'hpatches_cache' / 'hpatches_pairs.npz'}[MODE]
label = {'coco': 'COCO_synthetic', 'hpatches': 'HPatches'}[MODE]
t0 = time.time(); pairs = f.read_pair_cache(path)[START:END]; rows = []
for ix, p in enumerate(pairs):
    a = np.ascontiguousarray(p['d1'], np.float32); b = np.ascontiguousarray(p['d2'], np.float32)
    scene = p['name'].split('/')[0]
    cat = 'illumination' if scene.startswith('i_') else 'viewpoint' if scene.startswith('v_') else 'synthetic'

    def record(rot, M, K, dist, ij):
        i, j = ij
        err = np.linalg.norm(f.project(p['k1'][i], p['H']) - p['k2'][j], axis=1)
        rows.append(dict(dataset=label, category=cat, pair=p['name'], rotation=rot, M=M, K=K, distance=dist,
                         bytes=(M * int(np.log2(K)) // 8 if K else 256), mutual=len(i), correct_1px=int((err < 1).sum()),
                         correct_3px=int((err < 3).sum()), correct_5px=int((err < 5).sum())))

    record('none', 0, 0, 'float_cosine', mnn_from(-(a @ b.T)))
    for rot in ROT:
        mean, V = B[f'{rot}_mean'], B[f'{rot}_V']
        for M, K in CONFIGS:
            pm = perm_for(rot, M); books = [B[f'{rot}_{M}_{K}_{s}'] for s in range(M)]
            za, zb = ((a - mean) @ V)[:, pm], ((b - mean) @ V)[:, pm]
            ra, rb = decode(encode(za, books), books), decode(encode(zb, books), books)
            # cosine is taken in the original (uncentred) space: undo permutation, rotation and centring
            inv = np.argsort(pm)
            xa, xb = ra[:, inv] @ V.T + mean, rb[:, inv] @ V.T + mean
            record(rot, M, K, 'sdc_cos', mnn_from(-(unit(xa) @ unit(xb).T)))
            record(rot, M, K, 'sdc_l2', mnn_from((xa * xa).sum(1)[:, None] - 2 * xa @ xb.T + (xb * xb).sum(1)[None]))
            record(rot, M, K, 'adc_cos', mnn_from(-(a @ unit(xb).T)))
    if ix % 100 == 0: print(label, ix + 1, len(pairs), round(time.time() - t0), 's', flush=True)
pd.DataFrame(rows).to_csv(OUT / f'per_pair_{MODE}_{START:04d}.csv.gz', index=False)
print('done', label, round(time.time() - t0), 's')
