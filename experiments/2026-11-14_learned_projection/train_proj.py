"""2026-11-14 learned projection + thresholds: train a linear map W (64 -> m) and per-dimension thresholds jointly.

FIXED  : frozen XFeat descriptors; training / development pairs of 2026-11-14_learned_thresholds (disjoint from
         calibration and validation images); code rule count(z > edge); selection on development pairs only.
START  : W = top-m PCA directions of the calibration descriptors, thresholds = quantile edges of the projected
         calibration descriptors (this start is the `pca + quantize` baseline of 2026-11-14_dim_reduction).
CHANGED: W and the thresholds are updated by gradient descent on a matching loss (TRAIN=both), or only the
         thresholds (TRAIN=edges), or only W with thresholds re-fitted as quantiles of the calibration data (not used).
Relaxation: straight-through thermometer bits (hard forward, sigmoid backward, temperature = TAU_REL x per-dimension std
of the projected calibration data); distance = squared L2 between bin indices; symmetric cross-entropy loss.

usage: python train_proj.py CAL.npy PAIR_DIR ORIG_CODE_DIR OUT_DIR M LEVELS TRAIN(both|edges) LR_W STEPS [SEED]
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '2')
import sys, json, time
from pathlib import Path
import numpy as np
import jax, jax.numpy as jnp

CAL = np.load(sys.argv[1]).astype(np.float32); PD = Path(sys.argv[2]); ORIG = sys.argv[3]; OUT = Path(sys.argv[4])
M = int(sys.argv[5]); L = int(sys.argv[6]); TRAIN = sys.argv[7]; LR_W = float(sys.argv[8]); STEPS = int(sys.argv[9])
SEED = int(sys.argv[10]) if len(sys.argv) > 10 else 0
sys.path.insert(0, ORIG); import features as f
OUT.mkdir(parents=True, exist_ok=True)
TAG = f'm{M}_L{L}_{TRAIN}_lrw{LR_W}_s{SEED}'
TAU_REL, LR_U, LR_SCALE, EVAL_EVERY, NEG = 0.25, 0.02, 0.02, 300, 1e9

tr = np.load(PD / 'train_arrays.npz'); D1, D2, TGT, N2 = tr['d1'], tr['d2'], tr['tgt'].astype(np.int32), tr['n2'].astype(np.int32)
dev = f.read_pair_cache(PD / 'dev_pairs.npz')
MEAN = CAL.mean(0); Xc = CAL - MEAN
w, V = np.linalg.eigh(Xc.T @ Xc / len(Xc)); W0 = V[:, np.argsort(w)[::-1][:M]].astype(np.float32)
Z0 = Xc @ W0; SIG = Z0.std(0).astype(np.float32)
U0 = (np.quantile(Z0, np.arange(1, L) / L, axis=0).T / SIG[:, None]).astype(np.float32)      # thresholds in units of sigma
MEANj, SIGj = jnp.asarray(MEAN), jnp.asarray(SIG)


def index(x, W, u):
    z = ((x - MEANj) @ W)[:, :, None] - (u * SIGj[:, None])[None]
    s = jax.nn.sigmoid(z / (TAU_REL * SIGj[None, :, None])); h = (z > 0).astype(jnp.float32)
    return (s + jax.lax.stop_gradient(h - s)).sum(-1)


def loss_fn(params, a, b, tgt, n2):
    W, u, logs = params
    if TRAIN == 'edges': W = jax.lax.stop_gradient(W)
    ia, ib = index(a, W, u), index(b, W, u)
    logits = -((ia * ia).sum(1)[:, None] + (ib * ib).sum(1)[None] - 2 * ia @ ib.T) * jnp.exp(logs)
    logits = jnp.where(jnp.arange(b.shape[0])[None] < n2, logits, -NEG)
    wt = (tgt >= 0).astype(jnp.float32); j = jnp.maximum(tgt, 0)
    row = jax.nn.log_softmax(logits, axis=1)[jnp.arange(a.shape[0]), j]
    col = jax.nn.log_softmax(jnp.where(wt[:, None] > 0, logits, -NEG), axis=0)[jnp.arange(a.shape[0]), j]
    return -((row + col) * wt).sum() / (2 * wt.sum())


vg = jax.jit(jax.value_and_grad(loss_fn))


def codes(z, e):
    out = np.empty(z.shape, np.float32)
    for d in range(z.shape[1]): out[:, d] = np.searchsorted(e[d], z[:, d], side='left')
    return out


def dev_precision(W, u):
    e = u * SIG[:, None]; tot = cor = 0
    for p in dev:
        a, b = codes((p['d1'] - MEAN) @ W, e), codes((p['d2'] - MEAN) @ W, e)
        Dm = (a * a).sum(1)[:, None] + (b * b).sum(1)[None] - 2 * a @ b.T
        r = Dm.argmin(1); c = Dm.argmin(0); i = np.flatnonzero(c[r] == np.arange(len(r))); j = r[i]
        err = np.linalg.norm(f.project(p['k1'][i], p['H']) - p['k2'][j], axis=1); tot += len(i); cor += int((err < 3).sum())
    return 100 * cor / tot


ia0 = np.asarray(index(jnp.asarray(D1[0]), jnp.asarray(W0), jnp.asarray(U0))); ib0 = np.asarray(index(jnp.asarray(D2[0]), jnp.asarray(W0), jnp.asarray(U0)))
d0 = (ia0 * ia0).sum(1)[:, None] + (ib0 * ib0).sum(1)[None] - 2 * ia0 @ ib0.T
params = [jnp.asarray(W0), jnp.asarray(U0), jnp.asarray(np.log(10.0 / d0.std()), jnp.float32)]
m_ = [jnp.zeros_like(p) for p in params]; v_ = [jnp.zeros_like(p) for p in params]
rng = np.random.default_rng(SEED); order = rng.permutation(len(D1)); t0 = time.time()
log = [dict(step=0, dev_precision_3px=dev_precision(W0, U0), seconds=0.0)]
best = (log[0]['dev_precision_3px'], 0, W0.copy(), U0.copy()); run = 0.0
print(TAG, log[0], flush=True)
for step in range(1, STEPS + 1):
    k = order[(step - 1) % len(order)]
    if (step - 1) % len(order) == len(order) - 1: order = rng.permutation(len(D1))
    val, g = vg(params, jnp.asarray(D1[k]), jnp.asarray(D2[k]), jnp.asarray(TGT[k]), int(N2[k])); run += float(val)
    for i, lr in enumerate((LR_W, LR_U, LR_SCALE)):
        m_[i] = 0.9 * m_[i] + 0.1 * g[i]; v_[i] = 0.999 * v_[i] + 0.001 * g[i] ** 2
        params[i] = params[i] - lr * (m_[i] / (1 - 0.9 ** step)) / (jnp.sqrt(v_[i] / (1 - 0.999 ** step)) + 1e-8)
    params[1] = jnp.sort(params[1], axis=1)
    if step % EVAL_EVERY == 0 or step == STEPS:
        Wn, un = np.asarray(params[0]), np.asarray(params[1]); pr = dev_precision(Wn, un)
        log.append(dict(step=step, train_loss=run / EVAL_EVERY, dev_precision_3px=pr,
                        w_change=float(np.linalg.norm(Wn - W0) / np.linalg.norm(W0)), seconds=round(time.time() - t0, 1)))
        run = 0.0; print(log[-1], flush=True)
        if pr > best[0]: best = (pr, step, Wn.copy(), un.copy())
np.savez(OUT / f'model_{TAG}.npz', mean=MEAN, sigma=SIG, W_best=best[2], edges_best=best[3] * SIG[:, None], W_init=W0, edges_init=U0 * SIG[:, None])
(OUT / f'log_{TAG}.json').write_text(json.dumps(dict(tag=TAG, m=M, levels=L, train=TRAIN, lr_w=LR_W, lr_u=LR_U, tau_rel=TAU_REL, steps=STEPS, seed=SEED,
    bytes=M * int(np.log2(L)) // 8, best_dev=best[0], best_step=best[1], init_dev=log[0]['dev_precision_3px'], log=log, jax=jax.__version__), indent=1))
print('best dev', round(best[0], 3), 'at step', best[1], 'init', round(log[0]['dev_precision_3px'], 3))
