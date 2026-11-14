"""Convert train/dev pair caches into fixed-shape arrays with ground-truth correspondences (numpy only).
Ground truth: keypoint i of image 1 and j of image 2 are a positive pair when they are mutual nearest neighbours
under the homography (projected k1 vs k2) and closer than 3 px.
usage: python prep_pairs.py ORIG_CODE_DIR PAIR_DIR"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, sys.argv[1]); import features as f
D = Path(sys.argv[2])
for split in ('train', 'dev'):
    pairs = f.read_pair_cache(D / f'{split}_pairs.npz'); n = len(pairs)
    d1 = np.zeros((n, 1000, 64), np.float32); d2 = np.zeros_like(d1); tgt = np.full((n, 1000), -1, np.int16)
    n1 = np.zeros(n, np.int16); n2 = np.zeros(n, np.int16)
    for i, p in enumerate(pairs):
        a, b = len(p['d1']), len(p['d2']); d1[i, :a] = p['d1']; d2[i, :b] = p['d2']; n1[i], n2[i] = a, b
        g = np.linalg.norm(f.project(p['k1'], p['H'])[:, None] - p['k2'][None], axis=2)
        r = g.argmin(1); c = g.argmin(0); ok = (c[r] == np.arange(a)) & (g[np.arange(a), r] < 3)
        tgt[i, :a] = np.where(ok, r, -1)
    np.savez(D / f'{split}_arrays.npz', d1=d1, d2=d2, tgt=tgt, n1=n1, n2=n2)
    print(split, n, 'pairs; keypoints min', int(n1.min()), int(n2.min()), '; positives per pair mean', float((tgt >= 0).sum(1).mean()))
