"""2026-11-07 loss decomposition: quantization loss vs distance-function loss.

FIXED   : frozen XFeat descriptors/keypoints (regenerated 2026-10-14 cache),
          the ORIGINAL saved per-dimension quantile edges (fixed_quantile_edges.npz),
          exact mutual nearest neighbours, homography reprojection metric.
CHANGED : only the space/function in which the MNN distance is computed.

  A  float_cosine        : float descriptors, cosine (reference)
  B  recon_cosine        : bin -> calibration conditional mean per dim, renormalize, cosine
  C  recon_L2            : same reconstruction, no renormalization, squared L2
  D  bin_L1              : integer bin index L1 (= current thermometer XOR/popcount)
  E  bin_L2              : integer bin index squared L2
  D_tiebreak_float       : D, but ties broken by float cosine (diagnostic only, not deployable)

Reconstruction values are computed from the calibration descriptors only.
No setting is selected from these results (diagnostic, not tuning).
"""
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd

ORIG, CACHE, EDGES, OUT = map(Path, sys.argv[1:5])
BINS = [int(b) for b in sys.argv[5].split(',')] if len(sys.argv) > 5 else [8]
sys.path.insert(0, str(ORIG))
import features as f

OUT.mkdir(parents=True, exist_ok=True)
calibration = np.load(CACHE / 'coco' / 'calibration.npy')
edges_all = np.load(EDGES)


def codes(x, e):  # identical rule to optimized.ThermometerVariant.codes: count of x > edge
    return (x[:, :, None] > e[None]).sum(2).astype(np.int16)


def recon_table(e, bins):
    c = codes(calibration, e)
    table = np.zeros((64, bins), np.float32)
    for d in range(64):
        s = np.bincount(c[:, d], weights=calibration[:, d], minlength=bins)
        n = np.bincount(c[:, d], minlength=bins)
        assert (n > 0).all()
        table[d] = s / n
    return table


def mnn_min(D):  # argmin returns the smallest index on ties, same as the C++ kernels
    r = D.argmin(1); c = D.argmin(0)
    i = np.flatnonzero(c[r] == np.arange(len(r)))
    return i, r[i]


def l1_int(ca, cb):
    out = np.empty((len(ca), len(cb)), np.int32)
    for s in range(0, len(ca), 128):
        out[s:s + 128] = np.abs(ca[s:s + 128, None, :] - cb[None]).sum(2)
    return out


def sqdist(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T


def variants(a, b, e, table):
    ca, cb = codes(a, e), codes(b, e)
    ra = table[np.arange(64)[None], ca]; rb = table[np.arange(64)[None], cb]
    na = ra / np.linalg.norm(ra, axis=1, keepdims=True); nb = rb / np.linalg.norm(rb, axis=1, keepdims=True)
    cos = a @ b.T
    L1 = l1_int(ca, cb)
    caf, cbf = ca.astype(np.float64), cb.astype(np.float64)
    out = {
        'A_float_cosine': mnn_min(-cos),
        'B_recon_cosine': mnn_min(-(na @ nb.T)),
        'C_recon_L2': mnn_min(sqdist(ra, rb)),
        'D_bin_L1': mnn_min(L1),
        'E_bin_L2': mnn_min(sqdist(caf, cbf)),
        # (1-cos)/4 lies in [0, 0.5], so it never reorders different integer L1 values.
        'D_tiebreak_float': mnn_min(L1 + (1.0 - cos) / 4.0),
    }
    row_min = L1.min(1, keepdims=True); col_min = L1.min(0, keepdims=True)
    tie = dict(row_tie_frac=float(((L1 == row_min).sum(1) > 1).mean()),
               col_tie_frac=float(((L1 == col_min).sum(0) > 1).mean()))
    return out, tie


rows, ties = [], []
for b in BINS:
    e = edges_all[f'bins_{b}']
    table = recon_table(e, b)
    np.save(OUT / f'recon_table_{b}bin.npy', table)
    for label, path in [('COCO_synthetic', CACHE / 'coco' / 'test_pairs.npz'),
                        ('HPatches', CACHE / 'hpatches_cache' / 'hpatches_pairs.npz')]:
        pairs = f.read_pair_cache(path)
        for p in pairs:
            a, bb = p['d1'].astype(np.float32), p['d2'].astype(np.float32)
            res, tie = variants(a, bb, e, table)
            ref = set(zip(*map(tuple, res['A_float_cosine'])))
            scene = p['name'].split('/')[0]
            cat = 'illumination' if scene.startswith('i_') else 'viewpoint' if scene.startswith('v_') else 'synthetic'
            ties.append(dict(bins=b, dataset=label, category=cat, pair=p['name'], **tie))
            for name, (i, j) in res.items():
                err = np.linalg.norm(f.project(p['k1'][i], p['H']) - p['k2'][j], axis=1)
                got = set(zip(i.tolist(), j.tolist()))
                rows.append(dict(bins=b, dataset=label, category=cat, pair=p['name'], method=name,
                                 mutual=len(i), correct_1px=int((err < 1).sum()), correct_3px=int((err < 3).sum()),
                                 correct_5px=int((err < 5).sum()), same_as_float=len(got & ref)))
        print('done', b, label, flush=True)

df = pd.DataFrame(rows); df.to_csv(OUT / 'per_pair.csv', index=False)
tf = pd.DataFrame(ties); tf.to_csv(OUT / 'ties_per_pair.csv', index=False)


def summ(g):
    r = dict(pairs=len(g), mutual=int(g.mutual.sum()), correct_3px=int(g.correct_3px.sum()),
             same_as_float=int(g.same_as_float.sum()))
    for px in (1, 3, 5):
        r[f'precision_{px}px'] = 100 * g[f'correct_{px}px'].sum() / g.mutual.sum()
    return r


keys = ['bins', 'dataset', 'method']
summary = pd.DataFrame([dict(zip(keys, k), **summ(g)) for k, g in df.groupby(keys, sort=False)])
summary.to_csv(OUT / 'summary.csv', index=False)
keys2 = ['bins', 'dataset', 'category', 'method']
pd.DataFrame([dict(zip(keys2, k), **summ(g)) for k, g in df.groupby(keys2, sort=False)]).to_csv(OUT / 'by_type.csv', index=False)
tsum = tf.groupby(['bins', 'dataset', 'category'])[['row_tie_frac', 'col_tie_frac']].mean().reset_index()
tsum.to_csv(OUT / 'ties_summary.csv', index=False)
print(summary.round(3).to_string(index=False)); print(tsum.round(4).to_string(index=False))
