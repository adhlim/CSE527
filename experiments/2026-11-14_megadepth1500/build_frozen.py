"""Collect every parameter the 15 pre-registered configurations need into one frozen file, then check on the
validation caches that md_configs.match reproduces the validation numbers of the earlier experiments.
usage: python build_frozen.py CSE527_ROOT"""
import sys, json, hashlib
from pathlib import Path
import numpy as np
ROOT = Path(sys.argv[1]); E = ROOT / 'experiments'; HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / 'inputs' / 'orig_code'))
import md_configs as mc, features as f
CAL = np.load(ROOT / 'data/2026-10-14_regen_cache/coco/calibration.npy').astype(np.float32)


def table(z, e):
    L = e.shape[1] + 1; c = mc.codes(z, e); t = np.zeros((z.shape[1], L), np.float32)
    for d in range(z.shape[1]):
        n = np.bincount(c[:, d], minlength=L); s = np.bincount(c[:, d], weights=z[:, d], minlength=L)
        mids = np.concatenate([[e[d][0]], (e[d][:-1] + e[d][1:]) / 2, [e[d][-1]]]); t[d] = np.where(n > 0, s / np.maximum(n, 1), mids)
    return t


P = {}
ea = np.load(E / '2026-11-08_accuracy_matrix/results/edges_all.npz')
for k in ('percentile_16', 'quantile_8', 'quantile_4', 'uniform_4'): P[f'e_{k}'] = ea[k].astype(np.float32)
P['t_quantile_8'] = table(CAL, P['e_quantile_8'])
P['e_learned_4'] = np.load(E / '2026-11-14_learned_thresholds/results/edges_L4_l2_tau0.03_lr0.002_s0.npz')['best'].astype(np.float32)
P['int8_scale'] = np.float32(np.abs(CAL).max() / 127)
pr = np.load(E / '2026-11-14_dim_reduction/results/projections.npz'); P['mean'] = pr['mean'].astype(np.float32)
for k, W in (('pca32', pr['pca'][:, :32]), ('gfr32', pr['gfr'][:, :32]), ('pca48', pr['pca'][:, :48])):
    W = W.astype(np.float32); z = ((CAL - P['mean']) @ W).astype(np.float32)
    e = np.quantile(z, np.arange(1, 16) / 16, axis=0).T.astype(np.float32)
    P[f'W_{k}'] = W; P[f'e_{k}_16'] = e; P[f't_{k}_16'] = table(z, e)
cb = np.load(E / '2026-11-14_product_quantization/results/codebooks_pca64.npz')
P['pq_mean'] = cb['pca64_mean']; P['pq_V'] = cb['pca64_V']; P['pq_perm'] = np.arange(64).reshape(4, 16).T.ravel()
P['pq_inv'] = np.argsort(P['pq_perm']); P['pq_books'] = np.stack([cb[f'pca64_16_256_{s}'] for s in range(16)])
lp = np.load(E / '2026-11-14_learned_projection/results/model_m32_L16_both_lrw0.001_s0.npz')
P['lp_mean'] = lp['mean']; P['lp_W'] = lp['W_best']; P['lp_edges'] = lp['edges_best']
np.savez(HERE / 'frozen_configs.npz', **P)
sha = hashlib.sha256((HERE / 'frozen_configs.npz').read_bytes()).hexdigest(); print('frozen_configs.npz sha256', sha)

# ---- reproduce validation numbers ----
EXPECT = {'C0': (60.59, 70.57), 'C1': (60.58, 70.54), 'C2': (60.25, 70.05), 'C3': (58.51, 68.64), 'C4': (59.46, 69.34), 'C5': (59.55, 69.13),
          'C6': (57.59, 66.84), 'C7': (52.42, 60.40), 'C8': (57.51, 66.86), 'C9': (59.10, 69.54), 'C10': (59.11, 69.61), 'C11': (59.97, 70.05),
          'C12': (59.31, 68.94), 'C13': (58.01, 70.44), 'C14': (59.91, 70.51)}
P = mc.load(HERE / 'frozen_configs.npz'); which = sys.argv[2] if len(sys.argv) > 2 else 'both'; out = {}
for label, path in [('HPatches', ROOT / 'data/2026-10-14_regen_cache/hpatches_cache/hpatches_pairs.npz'), ('COCO', ROOT / 'data/2026-10-14_regen_cache/coco/test_pairs.npz')]:
    if which not in ('both', label): continue
    tot = {c: [0, 0] for c in mc.CONFIGS}
    for p in f.read_pair_cache(path):
        for c in mc.CONFIGS:
            i, j = mc.match(c, P, p['d1'], p['d2']); err = np.linalg.norm(f.project(p['k1'][i], p['H']) - p['k2'][j], axis=1)
            tot[c][0] += len(i); tot[c][1] += int((err < 3).sum())
    for c in mc.CONFIGS:
        v = 100 * tot[c][1] / tot[c][0]; exp = EXPECT[c][0 if label == 'HPatches' else 1]; out[f'{c}_{label}'] = v
        print(f"{label:9s} {c:4s} got {v:6.2f} expected {exp:6.2f} {'OK' if abs(v - exp) < 0.02 else 'DIFF'}  mutual {tot[c][0]} correct {tot[c][1]}", flush=True)
(HERE / 'results' / f'config_check_{which}.json').write_text(json.dumps(dict(sha256=sha, precision_3px=out), indent=1))
